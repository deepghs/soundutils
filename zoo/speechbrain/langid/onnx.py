import os
import tempfile

import onnx
import torch

from .model import make_sample_input, LangIdModel
from ...utils import onnx_optimize


def export_feat_model_to_onnx(model, onnx_filename, opset_version: int = 18, verbose: bool = True,
                              no_optimize: bool = False):
    w = make_sample_input()

    if torch.cuda.is_available():
        w = w.cuda()
        model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.dynamo_export(
            model,
            w,
            # verbose=verbose,
            # input_names=["wavs", "wav_lens"],
            # output_names=["preds"],
            #
            # opset_version=opset_version,
            # dynamic_axes={
            #     "wavs": {0: "batch", 1: "frames"},
            #     "wav_lens": {0: "batch"},
            #     "preds": {0: "batch"},
            # }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


if __name__ == '__main__':
    model = LangIdModel()
    export_feat_model_to_onnx(
        model,
        onnx_filename=os.path.join('mdist', 'langid.onnx')
    )
