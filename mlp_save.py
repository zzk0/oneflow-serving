import os
import unittest
import shutil
import numpy as np
import google.protobuf.text_format as text_format

import oneflow as flow
import oneflow.core.serving.saved_model_pb2 as saved_model_pb

from mlp_model import mlp_model


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def make_mlp_infer_func():
    input_lbns = {}
    output_lbns = {}

    @flow.global_function(type="predict")
    def mlp_inference(
            images: flow.typing.Numpy.Placeholder((1, 1, 28, 28), dtype=flow.float32),
            labels: flow.typing.Numpy.Placeholder((1,), dtype=flow.int32)
    ) -> flow.typing.Numpy:
        input_lbns["image"] = images.logical_blob_name
        input_lbns["label"] = labels.logical_blob_name
        with flow.scope.placement("gpu", "0:0"):
            logits = mlp_model(images, labels, train=False)
        output_lbns["output"] = logits.logical_blob_name
        return logits

    return mlp_inference, input_lbns, output_lbns


def save_model():
    init_env()
    mlp_infer, input_lbns, output_lbns = make_mlp_infer_func()
    print(mlp_infer.__name__)
    flow.load_variables(flow.checkpoint.get("./mlp_model"))

    saved_model_path = "models"
    model_name = "mlp"
    model_version = 1

    saved_model_builder = (
        flow.saved_model.ModelBuilder(saved_model_path)
        .ModelName(model_name)
        .Version(model_version)
    )
    saved_model_builder.AddFunction(mlp_infer).Finish()
    saved_model_builder.Save()

if __name__ == '__main__':
    save_model()
