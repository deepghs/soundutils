'optimum-cli export onnx --model'
import io
import os.path
import shutil
import subprocess
from typing import Optional

import click
import yaml
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_fs, upload_directory_as_directory, get_hf_client

_OP_CLI = shutil.which('optimum-cli')

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
def cli():
    pass


@cli.command('export', help='Export to onnx model',
             context_settings=GLOBAL_CONTEXT_SETTINGS)
@click.option('-r', '--repo_id', 'repo_id', type=str, required=True,
              help='Repository of original model.')
@click.option('-o', '--dst_repo_id', 'dst_repo_id', type=str, default=None,
              help='Destination repository for exported model.', show_default=True)
@click.option('--private', 'private', type=bool, is_flag=True, default=False,
              help='Use a private repository.', show_default=True)
def cli_export(repo_id: str, dst_repo_id: Optional[str], private: bool):
    logging.try_init_root(level=logging.INFO)

    dst_repo_id = dst_repo_id or f'deepghs/{repo_id.split("/")[-1]}-onnx'
    logging.info(f'Exporting {repo_id!r} --> {dst_repo_id!r} ...')

    if not _OP_CLI:
        raise EnvironmentError('No optimum-cli found in environment.')

    hf_client = get_hf_client()
    hf_fs = get_hf_fs()

    with TemporaryDirectory() as td:
        command = [_OP_CLI, 'export', 'onnx', '--model', repo_id, td]
        logging.info(f'Exporting with command {command!r} ...')
        ret = subprocess.run(command)
        ret.check_returncode()

        origin_readme_text = hf_fs.read_text(f'{repo_id}/README.md')
        with io.StringIO() as fm:
            is_started = False
            for line in origin_readme_text.splitlines(keepends=False):
                if line.strip() == '---':
                    if not is_started:
                        is_started = True
                    else:
                        break
                else:
                    print(line, file=fm)

            meta_text = fm.getvalue()

        with io.StringIO(meta_text) as fm:
            meta = yaml.safe_load(fm)
        meta['base_model'] = repo_id
        if 'tags' not in meta:
            meta['tags'] = []
        meta['tags'].append('audio')
        meta['tags'].append('onnx')
        meta['tags'].append('transformers')
        meta['library_name'] = 'transformers'

        with open(os.path.join(td, 'README.md'), 'w') as f:
            print('---', file=f)
            yaml.safe_dump(meta, f)
            print('---', file=f)
            print('', file=f)

            print(f'This is the ONNX exported version of '
                  f'[{repo_id}](https://huggingface.co/{repo_id}).', file=f)
            print('', file=f)

        if not hf_client.repo_exists(repo_id=dst_repo_id, repo_type='model'):
            logging.info(f'Creating repository {dst_repo_id!r} ...')
            hf_client.create_repo(repo_id=dst_repo_id, repo_type='model', private=private)
        upload_directory_as_directory(
            repo_id=dst_repo_id,
            repo_type='model',
            local_directory=td,
            path_in_repo='.',
            message=f'Export ONNX version of model {repo_id!r}'
        )


if __name__ == '__main__':
    cli()
