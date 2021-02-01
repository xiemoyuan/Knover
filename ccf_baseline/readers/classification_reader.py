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
"""Classification Reader."""

from collections import namedtuple

import numpy as np

from readers.dialog_reader import DialogReader
from utils import pad_batch_data
from utils.args import str2bool
import utils.tokenization as tokenization


class ClassificationReader(DialogReader):
    """Classification Reader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DialogReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(ClassificationReader, self).__init__(args)
        self.fields.append("label")
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))

        self.attention_style = args.attention_style
        self.mix_negative_sample = args.mix_negative_sample
        return

    def _convert_example_to_record(self, example, is_infer):
        field_values = self._parse_src(example.src)

        if self.max_knowledge_len > 0:
            knowledge_field_values = self._parse_knowledge(example.knowledge)
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

        if self.position_style == "continuous":
            field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["tgt_start_idx"] = tgt_start_idx
        field_values["data_id"] = example.data_id

        if not is_infer:
            field_values["label"] = int(example.label)

        record = self.Record(**field_values)
        return record

    def _read_numerical_file(self, fp, delimiter=";"):
        # Classification task does not support `numerical` data_format.
        raise NotImplementedError

    def _pad_batch_records(self, batch_records, is_infer):
        """
        Padding batch records and construct model's inputs.
        """
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=self.pad_id)
        if self.use_turn:
            batch_turn_ids = [record.turn_ids for record in batch_records]
            batch["turn_ids"] = pad_batch_data(batch_turn_ids, pad_id=self.pad_id)

        attention_mask = self._gen_self_attn_mask(batch_token_ids, is_unidirectional=False)
        batch["attention_mask"] = attention_mask
        max_len = max(map(len, batch_token_ids))
        batch["label_pos"] = np.arange(0, len(batch_records), dtype="int64").reshape([-1, 1]) * max_len

        batch_data_id = [record.data_id for record in batch_records]
        batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])

        if not is_infer:
            batch_label = [record.label for record in batch_records]
            batch["label"] = np.array(batch_label).astype("int64").reshape([-1, 1])
        return batch
