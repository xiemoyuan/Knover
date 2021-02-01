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
"""Classifier."""

import paddle.fluid as fluid
import paddle.fluid.layers as layers

from . import register_model
from .model_base import Model
from .unified_transformer import UnifiedTransformer
from utils.args import str2bool


@register_model("Classifier")
class Classifier(UnifiedTransformer):
    """Classifier."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = Model.add_cmdline_args(parser)
        group.add_argument("--weight_sharing", type=str2bool, default=True)
        group.add_argument("--mem_efficient", type=str2bool, default=False)
        group.add_argument("--use_role", type=str2bool, default=False)
        group.add_argument("--use_turn", type=str2bool, default=False)

        group.add_argument("--num_classes", type=int, default=2)

        return group

    def __init__(self, args, place):
        self.num_classes = args.num_classes
        super(Classifier, self).__init__(args, place)

    def _get_feed_dict(self, is_infer=False):
        """
        Get the feed list of the model.

        Args:
            is_infer(bool): True if running inference.

        Returns:
            dict(str, Variable): The feed dict.
        """
        feed_dict = {}
        feed_dict["token_ids"] = layers.data(name="token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["type_ids"] = layers.data(name="type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["pos_ids"] = layers.data(name="pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        if self.use_role:
            feed_dict["role_ids"] = layers.data(name="role_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        if self.use_turn:
            feed_dict["turn_ids"] = layers.data(name="turn_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        feed_dict["attention_mask"] = layers.data(
            name="attention_mask",
            shape=[-1, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)
        feed_dict["label_pos"] = layers.data(name="label_pos", shape=[-1, 1], dtype="int64")

        if not is_infer:
            feed_dict["label"] = layers.data(name="label", shape=[-1, 1], dtype="int64")

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
        return feed_dict

    def forward(self, inputs, is_infer=False):
        outputs = {}
        self.generation_caches = None
        outputs["enc_out"], outputs["checkpoints"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            turn_ids=inputs.get("turn_ids", None),
            generation_mask=inputs["attention_mask"]
        )
        return outputs

    def _get_metrics(self, inputs, outputs):
        metrics = {}
        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_pos"])
        cls_fc_out = layers.fc(
            input=pooled_out,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_fc.w_0", initializer=self.param_initializer),
            bias_attr="cls_fc.b_0")
        cls_loss, cls_softmax = layers.softmax_with_cross_entropy(
            logits=cls_fc_out, label=inputs["label"], return_softmax=True)

        cls_acc = layers.accuracy(cls_softmax, inputs["label"])
        mean_cls_loss = layers.mean(cls_loss)

        metrics["loss"] = mean_cls_loss
        metrics["cls_loss"] = mean_cls_loss
        metrics["cls_acc"] = cls_acc
        if self.num_classes == 2:
            pred = layers.argmax(cls_softmax, axis=1)
            metrics["stat_tp"] = layers.reduce_sum(
                layers.logical_and(pred == 1, inputs["label"] == 1).astype("float32")
            )
            metrics["stat_fp"] = layers.reduce_sum(
                layers.logical_and(pred == 1, inputs["label"] == 0).astype("float32")
            )
            metrics["stat_tn"] = layers.reduce_sum(
                layers.logical_and(pred == 0, inputs["label"] == 0).astype("float32")
            )
            metrics["stat_fn"] = layers.reduce_sum(
                layers.logical_and(pred == 0, inputs["label"] == 1).astype("float32")
            )
        return metrics

    def infer(self, inputs, outputs):
        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_pos"])
        cls_fc_out = layers.fc(
            input=pooled_out,
            size=2,
            param_attr=fluid.ParamAttr(
                name="cls_fc.w_0", initializer=self.param_initializer),
            bias_attr="cls_fc.b_0")
        scores = layers.softmax(cls_fc_out)
        predictions = {"scores": scores, "data_id": inputs["data_id"]}
        return predictions

    def infer_step(self, inputs):
        return Model.infer_step(self, inputs)
