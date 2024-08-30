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


def _get_files_for_lang(lang: str, char_id: str):
    df = _get_table_for_lang(lang)
    df = df[(df['channels'] == 1) & (df['time'] >= 5.0) & (df['sample_rate'] == 44100) & (df['char_id'] == char_id)]

    try:
        long_file = list(df[(df['time'] >= 12.5) & (df['time'] <= 18.0)].sample(1)['filename'])[0]
    except ValueError:
        long_file = None

    try:
        medium_file = list(df[(df['time'] >= 7.0) & (df['time'] <= 12.0)].sample(1)['filename'])[0]
    except ValueError:
        medium_file = None

    try:
        short_file = list(df[(df['time'] >= 4.0) & (df['time'] <= 6.5)].sample(1)['filename'])[0]
    except ValueError:
        short_file = None

    df = df[~df['filename'].isin({long_file, medium_file, short_file})]
    r_files = list(df.sample(3)['filename'])

    retval = []
    if short_file:
        retval.append(('short', short_file))
    if medium_file:
        retval.append(('medium', medium_file))
    if long_file:
        retval.append(('long', long_file))
    retval.extend([(f'sample_{i}', r_files[i]) for i in range(len(r_files))])

    return retval


def make_assets_for_char(char_id, char_name):
    for tag, filename in tqdm(_get_files_for_lang('jp', char_id)):
        _, ext = os.path.splitext(filename)
        dst_filename = get_testfile('assets', 'speakers', char_name, f'jp_{char_name}_{tag}{ext}')
        os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
        logging.info(f'Making {tag!r} for char {char_name!r} --> {dst_filename!r} ...')
        hf_tar_file_download(
            repo_id=f'deepghs/arknights_voices_jp',
            repo_type='dataset',
            archive_in_repo='voices.tar',
            file_in_archive=filename,
            local_file=dst_filename,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    df = _get_table_for_lang('jp')

    for char_id, char_name in tqdm([
        ('char_350_surtr', 'surtr'),
        ('char_102_texas', 'texas'),
        ('char_2014_nian', 'nian'),
        ('char_124_kroos', 'kroos'),
        ('char_4064_mlynar', 'mlynar')
    ]):
        logging.info(f'Making for char {char_name!r} ...')
        make_assets_for_char(char_id, char_name)
