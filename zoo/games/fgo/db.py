import mimetypes
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory, download_file_to_file
from hfutils.utils import number_to_tag, download_file, get_requests_session
from tqdm import tqdm

from soundutils.data import Sound


def sync(src_repo: str = 'deepghs/fgo_voices_index', dst_repo: str = 'deepghs/fgo_voices_jp'):
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    if not hf_client.repo_exists(repo_id=dst_repo, repo_type='dataset'):
        hf_client.create_repo(repo_id=dst_repo, repo_type='dataset', private=False)
        hf_client.update_repo_visibility(repo_id=dst_repo, repo_type='dataset', private=False)
        attr_lines = hf_fs.read_text(f'datasets/{dst_repo}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{dst_repo}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    df = pd.read_parquet(hf_client.hf_hub_download(
        repo_id=src_repo,
        repo_type='dataset',
        filename='table.parquet',
    )).replace(np.nan, None)
    session = get_requests_session(max_retries=5, timeout=10)

    if hf_fs.exists(f'datasets/{dst_repo}/table.parquet'):
        df_rows = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=dst_repo,
            repo_type='dataset',
            filename='table.parquet'
        )).replace(np.nan, None)
        rows = df_rows.to_dict('records')
        exist_ids = set(df_rows['id'])
    else:
        rows = []
        exist_ids = set()
    original_count = len(rows)

    with TemporaryDirectory() as upload_dir:
        tar_file = os.path.join(upload_dir, 'voices.tar')
        if hf_fs.exists(f'datasets/{dst_repo}/voices.tar'):
            download_file_to_file(
                repo_id=dst_repo,
                repo_type='dataset',
                file_in_repo='voices.tar',
                local_file=tar_file,
            )

        df_download = df[~df['id'].isin(exist_ids)]
        with TemporaryDirectory() as td, tarfile.open(tar_file, 'a:') as tar:
            tp = ThreadPoolExecutor(max_workers=16)
            pg = tqdm(total=len(df_download), desc=f'Download Batch')

            def _download(item, dst_file):
                try:
                    download_file(item['file_url'], dst_file, session=session)
                    _ = Sound.open(dst_file)
                except Exception as err:
                    logging.warn(f'File {os.path.basename(dst_file)!r} skipped due to download error: {err!r}')
                    if os.path.exists(dst_file):
                        os.remove(dst_file)
                finally:
                    pg.update()

            for item in df_download.to_dict('records'):
                _, ext = os.path.splitext(urlsplit(item['file_url']).filename)
                dst_filename = os.path.join(td, item['id'] + ext)
                tp.submit(_download, item, dst_filename)

            tp.shutdown(wait=True)

            for item in tqdm(df_download.to_dict('records'), desc='Adding'):
                _, ext = os.path.splitext(urlsplit(item['file_url']).filename)
                dst_filename = os.path.join(td, item['id'] + ext)
                if item['id'] in exist_ids:
                    logging.warning(f'Item {item["id"]} already exist, skipped.')
                elif os.path.exists(dst_filename):
                    # logging.info(f'Adding file {item["id"]!r} ...')
                    filename = item['id'] + ext
                    tar.add(dst_filename, filename)
                    mimetype, _ = mimetypes.guess_type(dst_filename)
                    sound = Sound.open(dst_filename)
                    rows.append({
                        **item,
                        'time': sound.time,
                        'sample_rate': sound.sample_rate,
                        'frames': sound.samples,
                        'channels': sound.channels,
                        'filename': filename,
                        'mimetype': mimetype,
                        'file_size': os.path.getsize(dst_filename),
                    })
                    exist_ids.add(item['id'])

                    if os.path.exists(dst_filename):
                        os.remove(dst_filename)
                else:
                    logging.warning(f'Voice file {item["id"]!r} not found, skipped.')

        parquet_file = os.path.join(upload_dir, 'table.parquet')
        df_rows = pd.DataFrame(rows)
        df_rows = df_rows.sort_values(by=['char_id', 'id'], ascending=True)
        df_rows.to_parquet(parquet_file, engine='pyarrow', index=False)

        tar_create_index_for_directory(upload_dir)

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('task_categories:', file=f)
            print('- automatic-speech-recognition', file=f)
            print('- audio-classification', file=f)
            print('language:', file=f)
            print(f'- ja', file=f)
            print('tags:', file=f)
            print('- audio', file=f)
            print('- text', file=f)
            print('- voice', file=f)
            print('- anime', file=f)
            print('- fgo', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df_rows))}', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# JP Voice-Text Dataset for FGO Waifus', file=f)
            print(f'', file=f)
            print(f'This is the JP voice-text dataset for FGO playable characters. '
                  f'Very useful for fine-tuning or evaluating ASR/ASV models.', file=f)
            print(f'', file=f)
            print(f'Only the voices with strictly one voice actor is maintained here to '
                  f'reduce the noise of this dataset.', file=f)
            print(f'', file=f)

            print(f'{plural_word(len(df_rows), "record")}, '
                  f'{df_rows["time"].sum() / 60 / 60:.3g} hours in total. '
                  f'Average duration is approximately {df_rows["time"].mean():.3g}s.', file=f)
            print(f'', file=f)
            df_shown = df_rows[:50][
                ['id', 'char_id', 'voice_actor_name', 'voice_title', 'voice_text',
                 'time', 'sample_rate', 'file_size', 'filename', 'mimetype', 'file_url']
            ]
            print(df_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=dst_repo,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df_rows) - original_count, "new record")} for fgo jp voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        src_repo='deepghs/fgo_voices_index',
        dst_repo='deepghs/fgo_voices_jp',
    )
