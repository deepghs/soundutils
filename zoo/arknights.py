import mimetypes
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_create_index_for_directory
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory, download_file_to_file
from hfutils.utils import number_to_tag, download_file, get_requests_session
from tqdm import tqdm

from soundutils.data import Sound


@lru_cache()
def get_raw_meta(lang):
    if lang == 'zh':
        resp = requests.get(
            'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/excel/charword_table.json')
    else:
        prefix = {
            'jp': 'ja_JP',
            'en': 'en_US',
            'kr': 'ko_KR',
        }[lang]
        resp = requests.get(
            f'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData_YoStar/main/{prefix}/gamedata/excel/charword_table.json')

    return resp.json()


@lru_cache()
def get_cv_list(lang):
    d = get_raw_meta('zh')
    mt = {
        'zh': 'CN_MANDARIN',
        'en': 'EN',
        'jp': 'JP',
        'kr': 'KR',
    }

    for key, value in d["voiceLangDict"].items():
        if value['charId'] == key and key.startswith('char_') and mt[lang] in value['dict']:
            yield key, value['dict'][mt[lang]]['cvName'][0]


@lru_cache()
def get_text_for_lang(lang):
    d = get_raw_meta(lang)
    cvmap = {key: cvname for key, cvname in get_cv_list(lang)}
    rows = []
    vmap = {
        'jp': 'voice',
        'en': 'voice_en',
        'zh': 'voice_cn',
        'kr': 'voice_kr',
    }
    for key, value in d["charWords"].items():
        if value['charId'] in cvmap:
            rows.append({
                'id': key,
                'char_id': value['charId'],
                'voice_actor_name': cvmap[value['charId']],
                'char_word_id': value['charWordId'],
                'lock_description': value['lockDescription'],
                'place_type': value['placeType'],
                'unlock_param': value['unlockParam'],
                'unlock_type': value['unlockType'],
                'voice_asset': value['voiceAsset'],
                'voice_id': value['voiceId'],
                'voice_index': value['voiceIndex'],
                'voice_text': value['voiceText'],
                'voice_title': value['voiceTitle'],
                'voice_type': value['voiceType'],
                'word_key': value['wordKey'],
                'file_url': f'https://torappu.prts.wiki/assets/audio/{vmap[lang]}/{value["charId"]}/{value["voiceId"].lower()}.mp3',
            })

    return pd.DataFrame(rows)


def sync(lang):
    repository = f'deepghs/arknights_voices_{lang}'
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=False)
        hf_client.update_repo_visibility(repo_id=repository, repo_type='dataset', private=False)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    df = get_text_for_lang(lang)
    session = get_requests_session(max_retries=5, timeout=10)

    if hf_fs.exists(f'datasets/{repository}/table.parquet'):
        df_rows = pd.read_parquet(hf_client.hf_hub_download(
            repo_id=repository,
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
        if hf_fs.exists(f'datasets/{repository}/voices.tar'):
            download_file_to_file(
                repo_id=repository,
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
                dst_filename = os.path.join(td, item['id'] + '.mp3')
                tp.submit(_download, item, dst_filename)

            tp.shutdown(wait=True)

            for item in tqdm(df_download.to_dict('records'), desc='Adding'):
                dst_filename = os.path.join(td, item['id'] + '.mp3')
                if os.path.exists(dst_filename):
                    logging.info(f'Adding file {item["id"]!r} ...')
                    tar.add(dst_filename, item['id'] + '.mp3')
                    mimetype, _ = mimetypes.guess_type(dst_filename)
                    sound = Sound.open(dst_filename)
                    rows.append({
                        **item,
                        'time': sound.time,
                        'sample_rate': sound.sample_rate,
                        'frames': sound.samples,
                        'filename': item['id'] + '.mp3',
                        'mimetype': mimetype,
                        'file_size': os.path.getsize(dst_filename),
                    })
                    exist_ids.add(item['id'])
                else:
                    logging.warning(f'Voice file {item["id"]!r} not found, skipped.')

        parquet_file = os.path.join(upload_dir, 'table.parquet')
        df_rows = pd.DataFrame(rows)
        df_rows = df_rows.sort_values(by=['voice_actor_name', 'char_id', 'id'], ascending=True)
        df_rows.to_parquet(parquet_file, engine='pyarrow', index=False)

        tar_create_index_for_directory(upload_dir)

        with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('task_categories:', file=f)
            print('- automatic-speech-recognition', file=f)
            print('- audio-classification', file=f)
            print('language:', file=f)
            print(f'- {lang}', file=f)
            print('tags:', file=f)
            print('- voice', file=f)
            print('- anime', file=f)
            print('modality:', file=f)
            print('- audio', file=f)
            print('- text', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df_rows))}', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# {lang.upper()} Voice-Text Dataset for Arknights Waifus', file=f)
            print(f'', file=f)
            print(f'This is the {lang.upper()} voice-text dataset for arknights playable characters. '
                  f'Very useful for fine-tuning or evaluating ASR/ASV models.', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_rows), "record")} in total.', file=f)
            print(f'', file=f)
            df_shown = df_rows[:50][
                ['id', 'char_id', 'voice_actor_name', 'voice_title', 'voice_text',
                 'time', 'sample_rate', 'file_size', 'filename', 'mimetype', 'file_url']
            ]
            print(df_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df_rows) - original_count, "new record")} for arknights {lang} voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        lang=os.environ.get('AK_LANG', 'jp'),
    )
