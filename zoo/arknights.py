import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        hf_client.update_repo_visibility(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    df = get_text_for_lang(lang)
    session = get_requests_session(max_retries=5, timeout=10.0)

    if hf_fs.exists(f'datasets/{repository}/exist_ids.json'):
        exist_ids = set(json.loads(hf_fs.read_text(f'datasets/{repository}/exist_ids.json')))
    else:
        exist_ids = set()

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

            for item in df_download.to_dict('records'):
                dst_filename = os.path.join(td, item['id'] + '.mp3')
                if os.path.exists(dst_filename):
                    logging.info(f'Adding file {item["id"]!r} ...')
                    tar.add(dst_filename, item['id'] + '.mp3')
                    exist_ids.add(item['id'])
                else:
                    logging.warning(f'Voice file {item["id"]!r} not found, skipped.')

        parquet_file = os.path.join(upload_dir, 'table.parquet')
        df.to_parquet(parquet_file, engine='pyarrow', index=False)

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
            print(f'- {number_to_tag(len(df))}', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'# {lang.upper()} Voice-Text Dataset for Arknights Waifus', file=f)
            print(f'', file=f)
            print(f'This is the {lang.upper()} voice-text dataset for arknights playable characters. '
                  f'Very useful for fine-tuning or evaluating ASR/ASV models.', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df), "record")} in total.', file=f)
            print(f'', file=f)
            df_shown = df[:50][
                ['id', 'char_id', 'voice_actor_name', 'voice_title', 'voice_text', 'file_url']
            ]
            print(df_shown.to_markdown(index=False), file=f)
            print(f'', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df), "record")} for arknights {lang} voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        lang='jp',
    )
