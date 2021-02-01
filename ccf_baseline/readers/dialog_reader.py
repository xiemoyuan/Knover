#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dialogue Reader."""

import csv
from collections import namedtuple
from contextlib import contextmanager
import gzip
import os

import numpy as np 
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet

from utils import pad_batch_data
from utils.args import str2bool
from utils.masking import mask
import utils.tokenization as tokenization


class DialogReader(object):
    """The implement of DialogReader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = parser.add_argument_group("Reader")
        group.add_argument("--max_src_len", type=int, default=384)
        group.add_argument("--max_tgt_len", type=int, default=128)
        group.add_argument("--max_seq_len", type=int, default=512)
        group.add_argument("--max_knowledge_len", type=int, default=0)
        group.add_argument("--knowledge_position", type=str, default="post_src",
                           choices=["post_src", "pre_src"])
        group.add_argument("--truncate_first_turn", type=str2bool, default=False)
        group.add_argument("--file_format", type=str, default="file",
                           choices=["file", "filelist"])
        group.add_argument("--data_format", type=str, default="raw",
                           choices=["raw", "tokenized", "numerical"])
        group.add_argument("--in_tokens", type=str2bool, default=False)
        group.add_argument("--batch_size", type=int, default=16)
        group.add_argument("--position_style", type=str, default="continuous",
                           choices=["continuous", "relative"])
        group.add_argument("--random_seed", type=int, default=11)
        group.add_argument("--sort_pool_size", type=int, default=2 ** 16)

        group = parser.add_argument_group("Tokenizer")
        group.add_argument("--tokenizer", type=str, default="SentencePieceTokenizer")
        args, _ = parser.parse_known_args()
        tokenizer_cls = getattr(tokenization, args.tokenizer)
        tokenizer_cls.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        tokenizer_cls = getattr(tokenization, args.tokenizer)
        self.tokenizer = tokenizer_cls(args)
        self.vocab = self.tokenizer.vocab
        self.pad_id = args.pad_id = self.vocab["[PAD]"]
        self.bos_id = args.bos_id = self.vocab["[CLS]"]
        self.eos_id = args.eos_id = self.vocab["[SEP]"]
        self.unk_id = args.unk_id = self.vocab["[UNK]"]
        self.mask_id = args.mask_id = self.vocab["[MASK]"]
        self.vocab_size = args.get("vocab_size", 0)
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.max_seq_len = args.max_seq_len
        self.max_knowledge_len = args.max_knowledge_len
        self.knowledge_position = args.knowledge_position
        self.truncate_first_turn = args.truncate_first_turn
        self.file_format = args.file_format
        self.data_format = args.data_format
        self.in_tokens = args.in_tokens
        self.batch_size = args.batch_size
        self.position_style = args.position_style
        self.sort_pool_size = args.sort_pool_size

        assert args.max_knowledge_len + args.max_src_len + args.max_tgt_len <= args.max_seq_len, \
            "args.max_knowledge_len + args.max_src_len + args.max_tgt_len > args.max_seq_len"

        # random_seed must be set for data slicing when using multi-gpu
        self.global_rng = np.random.RandomState(args.random_seed)

        # training progress
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        # model related
        self.use_role = args.use_role
        self.use_turn = args.use_turn

        self.fields = ["token_ids", "type_ids", "pos_ids"]
        if args.use_role:
            self.fields.append("role_ids")
        if args.use_turn:
            self.fields.append("turn_ids")
        self.num_numerical_fields = len(self.fields)
        self.fields += ["tgt_start_idx", "data_id"]
        self.sort_key = lambda record: [len(record.token_ids)]

        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))

        self.features = {}
        return

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_epoch, self.current_file_index, self.total_file

    def _parse_src(self, src):
        # process src
        src_token_ids = []
        src_pos_ids = []
        if self.use_role:
            src_role_ids = []
            role_id_list = []
        if self.use_turn:
            src_turn_ids = []

        # tokenize src
        s_token_ids_list = []
        for s in src.split("[SEP]"):
            s = tokenization.convert_to_unicode(s).strip()
            if self.use_role:
                s, role_id = s.split("\1")
                role_id = int(role_id)
                role_id_list.append(role_id)

            if self.data_format == "tokenized":
                s_tokens = s.split(" ")
            else:
                s_tokens = self.tokenizer.tokenize(s)

            s_token_ids = self.tokenizer.convert_tokens_to_ids(s_tokens) + [self.eos_id]
            s_token_ids_list.append(s_token_ids)

        # trim src
        idx = len(s_token_ids_list) - 1
        total_token_num = 1
        while idx >= 0:
            total_token_num += len(s_token_ids_list[idx])
            if total_token_num > self.max_src_len:
                if self.truncate_first_turn and idx == 0:
                    truncated_ids = s_token_ids_list[idx][:self.max_src_len - total_token_num]
                    if len(truncated_ids) > 1:
                        s_token_ids_list[idx] = truncated_ids[:-1] + [self.eos_id]
                        idx -= 1
                break
            idx -= 1

        for i, s_token_ids in enumerate(s_token_ids_list[idx + 1:], idx + 1):
            src_token_ids += s_token_ids
            src_pos_ids += list(range(1, len(s_token_ids) + 1))
            if self.use_role:
                src_role_ids += [role_id_list[i]] * len(s_token_ids)
            if self.use_turn:
                src_turn_ids += [len(s_token_ids_list) - i] * len(s_token_ids)

        field_values = {
            "token_ids": [self.bos_id] + src_token_ids,
            "type_ids": [0] * (len(src_token_ids) + 1),
            "pos_ids": [0] + src_pos_ids
        }
        if self.use_role:
            field_values["role_ids"] = [0] + src_role_ids
        if self.use_turn:
            field_values["turn_ids"] = [len(s_token_ids_list) - idx - 1] + src_turn_ids

        for k in field_values:
            assert len(field_values[k]) == len(field_values["token_ids"]), \
                f"len(field_values[{k}]) != len(field_values['token_ids'])"
        return field_values

    def _parse_knowledge(self, knowledge):
        ks_token_ids = [self.bos_id]
        ks_pos_ids = [0]
        for k in knowledge.split("[SEP]"):
            k = tokenization.convert_to_unicode(k).strip()
            if self.data_format == "tokenized":
                k_tokens = k.split(" ")
            else:
                k_tokens = self.tokenizer.tokenize(k)

            k_token_ids = self.tokenizer.convert_tokens_to_ids(k_tokens) + [self.eos_id]
            ks_token_ids += k_token_ids
            ks_pos_ids += list(range(1, len(k_token_ids) + 1))

        ks_token_ids = ks_token_ids[:self.max_knowledge_len]
        ks_pos_ids = ks_pos_ids[:self.max_knowledge_len]

        field_values = {
            "token_ids": ks_token_ids,
            "type_ids": [2] * len(ks_token_ids),
            "pos_ids": ks_pos_ids
        }
        if self.use_role:
            field_values["role_ids"] = [0] * len(ks_token_ids)
        if self.use_turn:
            field_values["turn_ids"] = [0] * len(ks_token_ids)

        return field_values

    def _parse_tgt(self, tgt):
        # process tgt
        tgt = tokenization.convert_to_unicode(tgt).strip()
        if self.use_role:
            tgt, role_id = tgt.split("\1")
            role_id = int(role_id)
        if self.data_format == "tokenized":
            tgt_tokens = tgt.split(" ")
        else:
            tgt_tokens = self.tokenizer.tokenize(tgt)

        tgt_token_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
        tgt_token_ids.append(self.eos_id)

        # trim tgt
        tgt_token_ids = tgt_token_ids[:self.max_tgt_len - 1]

        tgt_token_ids = [self.bos_id] + tgt_token_ids

        field_values = {
            "token_ids": tgt_token_ids,
            "type_ids": [1] * len(tgt_token_ids),
            "pos_ids": list(range(len(tgt_token_ids)))
        }
        if self.use_role:
            field_values["role_ids"] = [role_id] * len(tgt_token_ids)
        if self.use_turn:
            field_values["turn_ids"] = [0] * len(tgt_token_ids)

        return field_values

    def _convert_example_to_record(self, example, is_infer):
        field_values = self._parse_src(example.src)

        if self.max_knowledge_len > 0:
            # add knowledge
            knowledge_field_values = self._parse_knowledge(example.knowledge)
            if self.knowledge_position == "pre_src":
                field_values = {
                    k: knowledge_field_values[k] + field_values[k]
                    for k in field_values
                }
            else:
                field_values = {
                    k: field_values[k] + knowledge_field_values[k]
                    for k in field_values
                }

        tgt_start_idx = len(field_values["token_ids"])

        if self.position_style == "relative":
            ctx_len = len(field_values["token_ids"])
            field_values["pos_ids"] = [
                self.max_tgt_len + ctx_len - i - 1
                for i in range(ctx_len)
            ]

        if not is_infer:
            tgt_field_values = self._parse_tgt(example.tgt)
            field_values = {
                k: field_values[k] + tgt_field_values[k]
                for k in field_values
            }

        if self.position_style == "continuous":
            field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["tgt_start_idx"] = tgt_start_idx
        field_values["data_id"] = example.data_id

        record = self.Record(**field_values)
        return record

    def _read_tsv(self, fp, phase, is_infer, delimiter="\t", quotechar=None):
        """Reads a tab separated value file."""
        csv.field_size_limit(2 ** 20)
        reader = csv.reader(fp, delimiter=delimiter, quotechar=quotechar)
        headers = next(reader)
        headers.append("data_id")
        Example = namedtuple("Example", headers)

        for i, line in enumerate(reader):
            example = Example(*line, data_id=i)
            if is_infer or phase.endswith("test"):
                self.features[i] = example
            record = self._convert_example_to_record(example, is_infer)
            yield record

    def _read_numerical_file(self, fp, delimiter=";", is_infer=False):
        for i, line in enumerate(fp):
            cols = tokenization.convert_to_unicode(line).strip().split(delimiter)
            cols = list(map(lambda x: list(map(int, x.split(" "))), cols))
            if len(cols) > self.num_numerical_fields:
                cols = cols[:self.num_numerical_fields]
            if not is_infer:
                tgt_start_idx = cols[0].index(self.bos_id, 1)
            else:
                tgt_start_idx = len(cols[0])
            record = self.Record(*cols, tgt_start_idx=tgt_start_idx, data_id=i)
            yield record

    def _read_file(self, input_file, phase, is_infer):
        def __wrapper__():
            with open_file(input_file) as fp:
                if self.data_format == "numerical":
                    records = self._read_numerical_file(fp, is_infer=is_infer)
                else:
                    records = self._read_tsv(fp, phase, is_infer)
                for record in records:
                    yield record

        return __wrapper__

    def _read_files(self, filelist, phase, is_infer, shuffle_files):
        input_files = open(filelist).readlines()
        def __wrapper__():
            if shuffle_files:
                self.global_rng.shuffle(input_files)

            if phase == "train":
                self.total_file = len(input_files)
            for file_index, input_file in enumerate(input_files, 1):
                if phase == "train":
                    self.current_file_index = file_index
                    self.current_file = input_file
                file_reader = self._read_file(input_file.strip(), phase, is_infer)
                for record in file_reader():
                    yield record

        return __wrapper__

    def _batch_reader(self, reader, phase=None, is_infer=False, sort_pool_size=2 ** 16):
        """Construct a batch reader."""
        def update_max_lens(max_lens, record):
            """Update max_lens."""
            if max_lens is None:
                return self.sort_key(record)
            else:
                return [max(max_len, l) for max_len, l in zip(max_lens, self.sort_key(record))]

        def get_batch(reader):
            """Generate batches from reader."""
            batch, max_lens = [], None
            for record in reader():
                if record is None:
                    if len(batch) > 0:
                        yield batch
                    batch, max_lens = [], None
                    continue

                self.current_example += 1
                max_lens = update_max_lens(max_lens, record)
                if self.in_tokens:
                    to_append = (len(batch) + 1) * sum(max_lens) <= self.batch_size
                else:
                    to_append = len(batch) < self.batch_size

                if to_append:
                    batch.append(record)
                else:
                    yield batch
                    batch, max_lens = [record], self.sort_key(record)

            if len(batch) > 0:
                yield batch

        def get_sorted_batch(pool):
            """Generate sorted batches from pool."""
            pool = sorted(pool, key=self.sort_key)
            batches = []
            batch, max_lens = [], None
            for record in pool:
                self.current_example += 1
                max_lens = update_max_lens(max_lens, record)
                if self.in_tokens:
                    to_append = (len(batch) + 1) * sum(max_lens) <= self.batch_size
                else:
                    to_append = len(batch) < self.batch_size
                if to_append:
                    batch.append(record)
                else:
                    batches.append(batch)
                    batch, max_lens = [record], self.sort_key(record)

            if len(batch) > 0:
                batches.append(batch)
            self.global_rng.shuffle(batches)

            for batch in batches:
                yield batch

        def __wrapper__():
            if sort_pool_size > 0:
                pool = []
                for record in reader():
                    if record is None:
                        continue
                    pool.append(record)
                    if len(pool) == sort_pool_size:
                        for batch in get_sorted_batch(pool):
                            yield batch
                        pool = []
                if len(pool) > 0:
                    for batch in get_sorted_batch(pool):
                        yield batch
            else:
                for batch in get_batch(reader):
                    yield batch

        return __wrapper__

    def _distributed_batch_reader(self, batch_reader, num_part, part_id, is_test=False):
        def __wrapper__():
            batches = []
            for batch in batch_reader():
                if batch is None:
                    continue
                batches.append(batch)
                if len(batches) == num_part:
                    yield batches[part_id]
                    batches = []
            if is_test and 0 <= part_id < len(batches):
                yield batches[part_id]
            return

        return __wrapper__

    def data_generator(self,
                       input_file=None,
                       reader=None,
                       num_epochs=1,
                       num_part=1,
                       part_id=0,
                       phase=None,
                       is_infer=False):
        """Data generator."""
        def __wrapper__():
            nonlocal reader
            if reader is None:
                if self.file_format == "filelist":
                    reader = self._read_files(input_file, phase, is_infer, not phase.endswith("test"))
                else:
                    if phase == "train":
                        self.total_file = 1
                        self.current_file_index = 1
                        self.current_file = input_file
                    reader = self._read_file(input_file, phase, is_infer)

            batch_reader = self._batch_reader(
                reader,
                phase,
                is_infer,
                sort_pool_size=self.sort_pool_size if not is_infer else 0)
            if phase == "train":
                batch_reader = self._distributed_batch_reader(batch_reader, num_part, part_id)
            elif phase.startswith("distributed"):
                batch_reader = self._distributed_batch_reader(batch_reader, num_part, part_id, is_test=True)

            for epoch_index in range(num_epochs):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index + 1
                for batch in batch_reader():
                    yield self._pad_batch_records(batch, is_infer)

        return __wrapper__

    def _gen_self_attn_mask(self, batch_token_ids, batch_tgt_start_idx=None, is_unidirectional=True, shift_len=0):
        max_len = max(map(len, batch_token_ids))
        input_mask_data = np.zeros((len(batch_token_ids), max_len + shift_len, max_len + shift_len))
        if is_unidirectional:
            for index, mask_data in enumerate(input_mask_data):
                start = 0 if batch_tgt_start_idx is None else batch_tgt_start_idx[index]
                end = len(batch_token_ids[index])
                mask_data[:end + shift_len, :start + shift_len] = 1.0
                # Generate the lower triangular matrix using the slice of matrix
                b = np.tril(np.ones([end - start, end - start]), 0)
                mask_data[start + shift_len:end + shift_len, start + shift_len:end + shift_len] = b
        else:
            for index, token_ids in enumerate(batch_token_ids):
                input_mask_data[index, :len(token_ids) + shift_len, :len(token_ids) + shift_len] = 1.0
        return input_mask_data.astype("float32")

    def _pad_batch_records(self, batch_records, is_infer):
        """
        Padding batch records and construct model's inputs.
        """
        batch_size = len(batch_records)
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
        if self.use_turn:
            batch_turn_ids = [record.turn_ids for record in batch_records]
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)
        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=self.pad_id)
        if self.use_turn:
            batch["turn_ids"] = pad_batch_data(batch_turn_ids, pad_id=self.pad_id)

        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        batch["generation_mask"] = self._gen_self_attn_mask(
            batch_token_ids,
            batch_tgt_start_idx=batch_tgt_start_idx)

        if is_infer:
            tgt_ids = np.array([[[self.bos_id]]] * len(batch_token_ids), dtype="int64")
            if self.position_style == "continuous":
                tgt_pos = np.array(batch_tgt_start_idx, dtype="int64")
            else:
                tgt_pos = np.zeros_like(batch_tgt_start_idx, dtype="int64")
            tgt_pos = tgt_pos.reshape(-1, 1, 1)
            batch["init_score"] = np.zeros_like(tgt_ids, dtype="float32").reshape(-1, 1).tolist()
            batch["tgt_ids"] = tgt_ids.tolist()
            batch["tgt_pos"] = tgt_pos.tolist()
            batch["parent_idx"] = np.array(range(batch_size), dtype="int32")

            batch["tgt_generation_mask"] = batch["generation_mask"][:, 0:1, :].astype("float32")
        else:
            batch["tgt_label"], batch["tgt_pos"] = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                sent_b_starts=batch_tgt_start_idx,
                is_unidirectional=True)

        batch_data_id = [record.data_id for record in batch_records]
        batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])
        return batch


@contextmanager
def open_file(filename):
    """Open file."""
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()
