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
"""Interactive server."""

import argparse
from collections import namedtuple
import json
import os
import random

import flask
from termcolor import colored, cprint
import paddle.fluid as fluid

import models
from tasks.dialog_generation import DialogGeneration
from utils import check_cuda
from utils.args import parse_args


def setup_args():
    """
    Setup arguments.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Server")
    group.add_argument("--port", type=int, default=8233)
    group.add_argument("--bot_name", type=str, default="Knover")

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    print(json.dumps(args, indent=2))
    return args


def run_server(args):
    """
    Inference main function.
    """
    dev_count = 1
    gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    task.debug()

    Example = namedtuple("Example", ["src", "knowledge", "data_id"])

    app = flask.Flask("interactive server")
    app.config["JSON_AS_ASCII"] = False

    @app.route("/api/chitchat", methods=["POST"])
    def chitchat():
        """Chitchat port."""
        req = flask.request.get_json(force=True)
        data_id = random.randint(0, 2 ** 31 - 1)
        example = Example(
            src=" [SEP] ".join(req["context"]),
            knowledge=" [SEP] ".join(req["knowledge"]),
            data_id=data_id)
        task.reader.features[data_id] = example
        record = task.reader._convert_example_to_record(example, is_infer=True)
        data = task.reader._pad_batch_records([record], is_infer=True)
        pred = task.infer_step(model, data)[0]
        bot_response = pred["response"]
        print(colored("[Bot]:", "blue", attrs=["bold"]), colored(bot_response, attrs=["bold"]))
        ret = {
            "name": args.bot_name,
            "reply": bot_response,
            "config": json.dumps(args)
        }
        return flask.jsonify(ret)

    os.system("echo http://`hostname -i`:{}/api/chitchat".format(args.port))
    app.run(host="0.0.0.0", port=args.port, debug=False)
    return


if __name__ == "__main__":
    args = setup_args()
    check_cuda(True)
    run_server(args)
