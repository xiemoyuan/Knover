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
"""Next sentence prediction task."""

import math

from readers.nsp_reader import NSPReader
from tasks import register_task
from tasks.task_base import Task
from utils.args import str2bool


@register_task("NextSentencePrediction")
class NextSentencePrediction(Task):
    """
    Define dialogue response generation.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = NSPReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(NextSentencePrediction, self).__init__(args)
        self.reader = NSPReader(args)
        return

    def merge_mertrics_and_statistics(self, outputs, part_outputs):
        """
        Merge two evaulation output.
        """
        if outputs is None:
            return part_outputs

        if part_outputs is None:
            return outputs

        batch_size = outputs.pop("batch_size")
        tokens_num = outputs.pop("tokens_num")
        part_batch_size = part_outputs.pop("batch_size")
        part_tokens_num = part_outputs.pop("tokens_num")

        new_outputs = {
            "batch_size": batch_size + part_batch_size,
            "tokens_num": tokens_num + part_tokens_num
        }
        for k in outputs:
            if k.startswith("token_"):
                new_outputs[k] = (
                    outputs[k] * tokens_num + part_outputs[k] * part_tokens_num
                ) / new_outputs["tokens_num"]
            else:
                new_outputs[k] = (
                    outputs[k] * batch_size + part_outputs[k] * part_batch_size
                ) / new_outputs["batch_size"]
        return new_outputs

    def get_metrics(self, outputs):
        """
        Get metrics.
        """
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        outputs.pop("batch_size", None)
        outputs.pop("tokens_num", None)
        metrics = {}
        for k in outputs:
            if k.startswith("token_"):
                metrics[k[6:]] = outputs[k]
            else:
                metrics[k] = outputs[k]
            if k == "token_lm_loss":
                metrics["ppl"] = math.exp(outputs[k])
        return metrics

    def _post_process_infer_output(self, predictions):
        predictions = [{"data_id": data_id.tolist()[0], "score": score.tolist()[1]}
                       for data_id, score in zip(predictions["data_id"], predictions["scores"])]
        return predictions
