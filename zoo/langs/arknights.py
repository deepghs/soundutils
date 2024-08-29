import os.path
from functools import lru_cache

import pandas as pd
from ditk import logging
from hfutils.index import hf_tar_file_download
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from test.testings import get_testfile

LANGS = ['jp', 'zh', 'kr', 'en']


@lru_cache()
def _get_table_for_lang(lang: str):
    df = pd.read_parquet(hf_hub_download(
        repo_id=f'deepghs/arknights_voices_{lang}',
        repo_type='dataset',
        filename='table.parquet',
    ))
    return df


def _get_files_for_lang(lang: str):
    df = _get_table_for_lang(lang)
    df = df[(df['channels'] == 1) & (df['time'] >= 5.0) & (df['sample_rate'] == 44100)]

    long_file = list(df[(df['time'] >= 15.0) & df['time'] <= 22.0].sample(1)['filename'])[0]
    medium_file = list(df[(df['time'] >= 9.0) & df['time'] <= 13.0].sample(1)['filename'])[0]
    short_file = list(df[(df['time'] >= 5.0) & df['time'] <= 8.0].sample(1)['filename'])[0]

    df = df[~df['filename'].isin({long_file, medium_file, short_file})]
    r_files = list(df.sample(3)['filename'])

    return [
        ('short', short_file),
        ('medium', medium_file),
        ('long', long_file),
        *[(f'sample_{i}', r_files[i]) for i in range(len(r_files))],
    ]


def make_assets_for_lang(lang):
    for tag, filename in tqdm(_get_files_for_lang(lang)):
        _, ext = os.path.splitext(filename)
        dst_filename = get_testfile('assets', 'langs', lang, f'{lang}_{tag}{ext}')
        os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
        logging.info(f'Making {tag!r} for language {lang!r} --> {dst_filename!r} ...')
        hf_tar_file_download(
            repo_id=f'deepghs/arknights_voices_{lang}',
            repo_type='dataset',
            archive_in_repo='voices.tar',
            file_in_archive=filename,
            local_file=dst_filename,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    for l in tqdm(LANGS):
        logging.info(f'Making for lang {l!r} ...')
        make_assets_for_lang(l)
