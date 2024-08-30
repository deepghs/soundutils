import os
import tempfile

import onnx
import torch
import yaml
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, upload_directory_as_directory
from pyannote.audio import Model

from soundutils.data import SoundTyping, Sound
from test.testings import get_testfile
from ...utils import onnx_optimize, get_readme_meta


def encode(sound: SoundTyping):
    sound = Sound.load(sound)
    sound = sound.resample(16000)
    data, sr = sound.to_numpy()
    input_ = torch.from_numpy(data).type(torch.float32)
    return input_


def export_embedding_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                                   no_optimize: bool = False):
    w = encode(get_testfile('assets', 'texas_long.wav'))
    model.eval()

    if torch.cuda.is_available():
        w = w.cuda()
        model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            w,
            onnx_model_file,
            do_constant_folding=True,
            verbose=verbose,
            input_names=["waveform"],
            output_names=["embeddings"],

            opset_version=opset_version,
            dynamic_axes={
                "waveform": {0: "batch", 1: "frames"},
                "embeddings": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


def sync(repo_id: str, src_repo_id: str = "pyannote/embedding"):
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
        hf_client.create_repo(repo_id=repo_id, repo_type='model')

    with TemporaryDirectory() as td:
        model = Model.from_pretrained(
            src_repo_id,
            use_auth_token=os.environ.get('HF_TOKEN')
        )
        export_embedding_model_to_onnx(model, os.path.join(td, 'model.onnx'))

        meta = get_readme_meta(src_repo_id)
        meta['base_model'] = src_repo_id
        if 'tags' not in meta:
            meta['tags'] = []
        if 'audio' not in meta['tags']:
            meta['tags'].append('audio')
        meta['tags'].append('onnx')

        with open(os.path.join(td, 'README.md'), 'w') as f:
            print('---', file=f)
            yaml.safe_dump(meta, f)
            print('---', file=f)

            print(f'This is the ONNX exported version of '
                  f'[{src_repo_id}](https://huggingface.co/{src_repo_id}).', file=f)
            print('', file=f)

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            local_directory=td,
            path_in_repo='.',
            message=f'Export ONNX version of model {src_repo_id!r}'
        )


if __name__ == '__main__':
    sync(
        repo_id='deepghs/pyannote-embedding-onnx',
        src_repo_id='pyannote/embedding',
    )
