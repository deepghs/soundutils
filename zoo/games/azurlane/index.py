import json
import os
import re
from typing import Optional
from urllib.parse import quote_plus, urljoin

import langdetect
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, get_hf_fs, get_hf_client
from hfutils.utils import get_requests_session, number_to_tag
from langdetect import LangDetectException
from pyquery import PyQuery as pq

__root_website_en__ = 'https://azurlane.koumakan.jp/'

from tqdm import tqdm


def _crawl_index_from_ensite(session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    response = session.get(f'{__root_website_en__}/wiki/List_of_Ships')
    records = {}
    page = pq(response.text)
    base_table, plan_table, meta_table, collab_table, _ = page('.wikitable').items()

    def _extract_row(row):
        link = row('td:nth-child(1) a')
        url = urljoin(__root_website_en__, link.attr('href'))
        assert len(urlsplit(url).path_segments) == 3
        assert urlsplit(url).path_segments[1] == 'wiki'
        name = urlsplit(url).path_segments[2]
        return {
            'name_link': name,
            'name': row('td:nth-child(2)').text().strip(),
            'rarity_v': int(re.fullmatch(r'rarity-(?P<r>\d+)', row('td:nth-child(3)').attr('class')).group('r')),
            'rarity': row('td:nth-child(3)').text().strip(),
            'class_v': re.fullmatch(r'class-(?P<c>[a-zA-Z\d]+)', row('td:nth-child(4)').attr('class')).group('c'),
            'class': row('td:nth-child(4)').text().strip(),
            'affiliation': row('td:nth-child(5)').text().strip()
            if row('td:nth-child(4)').attr('colspan') == '2' else row('td:nth-child(5)').text().strip(),
        }

    for row in base_table('tbody tr').items():
        link = row('td:nth-child(1) a')
        if link.text().strip():
            index = link.text().strip()
            records[index] = {
                'id': index,
                **_extract_row(row)
            }

    for row in plan_table('tbody tr').items():
        link = row('td:nth-child(1) a')
        if link.text().strip():
            index = f'Plan{link.text().strip()[-3:]}'
            records[index] = {
                'id': index,
                **_extract_row(row)
            }

    for row in meta_table('tbody tr').items():
        link = row('td:nth-child(1) a')
        if link.text().strip():
            index = f'META{link.text().strip()[-3:]}'
            records[index] = {
                'id': index,
                **_extract_row(row)
            }

    for row in collab_table('tbody tr').items():
        link = row('td:nth-child(1) a')
        if link.text().strip():
            index = f'Collab{link.text().strip()[-3:]}'
            records[index] = {
                'id': index,
                **_extract_row(row)
            }

    return records


def _get_info(link_name: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = session.get(f'https://azurlane.koumakan.jp/wiki/{quote_plus(link_name)}')
    resp.raise_for_status()

    page = pq(resp.text)
    card_content = page('.ship-card-content')
    headline = card_content('.card-headline')
    cnname = headline('span[lang=zh]').text().strip() or None
    jpname = headline('span[lang=ja]').text().strip() or None
    enname = headline('span[lang=en]').text().strip() or None

    va = None
    for row in card_content('.card-info table tr').items():
        if row('th:nth-child(1)').text().lower() == 'voice actor':
            ex = row('td:nth-child(2)')
            ex.remove('.sm2_button')
            va = ex.text().strip()
            break

    return {
        'cnname': cnname,
        'jpname': jpname,
        'enname': enname,
        'voice_actor_name': va,
    }


def _get_voices(link_name: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = session.get(f'https://azurlane.koumakan.jp/wiki/{quote_plus(link_name)}/Quotes')
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    page = pq(resp.text)
    retval = []
    for pannel in page('section.tabber__section > article.tabber__panel').items():
        if pannel.attr('data-title') and 'japanese' in pannel.attr('data-title').lower():
            for table in pannel('table').items():
                header_row = list(table('tr').items())[0]
                if len(list(header_row('th').items())) != 3:
                    logging.error(f'Table unrecognizable of {link_name!r}, skipped.')
                    continue

                group_name = list(table.prev_all('h3').items())[-1].text().strip()
                for row in table('tr').items():
                    if len(list(row('th').items())) == 1 and \
                            len(list(row('td').items())) == 2:
                        voice_title = row('th:nth-child(1)').text().strip()
                        voice_text = row('td:nth-child(3) span[lang=ja]').text().strip()
                        voice_text = re.sub(r'\([\s\S]*\)', '', voice_text).strip()
                        if not voice_text:
                            logging.warning(f'Voice {voice_title!r} text of {link_name!r} is empty, skipped.')
                            continue
                        # try:
                        #     if langdetect.detect(voice_text) != 'ja':
                        #         logging.warning(f'Voice text is not ja but {langdetect.detect(voice_text)!r} '
                        #                         f'of {link_name!r}, skipped - {voice_text!r}.')
                        #         continue
                        # except LangDetectException:
                        #     logging.warning(f'Voice text language is known '
                        #                     f'of {link_name!r}, skipped - {voice_text!r}.')
                        #     continue

                        if row('td:nth-child(2) a.sm2_button'):
                            file_url = urljoin(resp.url, row('td:nth-child(2) a.sm2_button').attr('href'))
                        else:
                            logging.warning(f'No file url found for voice {voice_title!r} of {link_name!r}, skipped.')
                            continue

                        retval.append({
                            'group': group_name,
                            'voice_title': voice_title,
                            'voice_text': voice_text,
                            'file_url': file_url,
                        })

    return retval


def sync(repository: str = f'deepghs/azurlane_voices_index', maxcnt: Optional[int] = None):
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

    session = get_requests_session()

    qs = list(_crawl_index_from_ensite(session=session).items())
    if maxcnt is not None:
        qs = qs[:maxcnt]
    d = []
    for index, chinfo in tqdm(qs):
        chinfo_extra = _get_info(link_name=chinfo['name_link'], session=session)
        voices = _get_voices(link_name=chinfo['name_link'], session=session)
        d.append({
            **chinfo,
            **chinfo_extra,
            'voices': voices,
        })

    with TemporaryDirectory() as upload_dir:
        with open(os.path.join(upload_dir, 'raw.json'), 'w') as f:
            json.dump(d, f, indent=4, sort_keys=True, ensure_ascii=False)

        rows = []
        for chinfo in d:
            for voiceinfo in chinfo['voices']:
                filename = urlsplit(voiceinfo['file_url']).filename
                voice_id, _ = os.path.splitext(filename)
                rows.append({
                    'id': f'char_{chinfo["id"]}_{voice_id}',
                    'char_id': chinfo['id'],
                    'char_name': chinfo['name'],
                    'char_jpname': chinfo['jpname'],
                    'char_cnname': chinfo['cnname'],
                    'char_enname': chinfo['enname'],
                    'char_rarity': chinfo['rarity'],
                    'char_rarity_v': chinfo['rarity_v'],
                    'char_class': chinfo['class'],
                    'char_class_v': chinfo['class_v'],
                    'char_affiliation': chinfo['affiliation'],
                    'voice_actor_name': chinfo['voice_actor_name'],
                    'voice_id': filename,
                    'voice_text': voiceinfo['voice_text'],
                    'voice_group': voiceinfo['group'],
                    'voice_title': voiceinfo['voice_title'],
                    'file_url': voiceinfo['file_url'],
                })
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(upload_dir, 'table.parquet'), engine='pyarrow', index=False)

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
            print('- azurlane', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df))}', file=f)
            print('---', file=f)
            print('', file=f)

            print('# Index Dataset for Azur Lane Voices', file=f)
            print('', file=f)
            print('This is the middleware index database for the Azur Lane playable characters.', file=f)
            print('', file=f)
            print(f'{plural_word(len(df), "record")} in total.', file=f)
            print('', file=f)
            df_shown = df[:50][
                ['id', 'char_id', 'char_name', 'voice_actor_name', 'voice_title', 'voice_text', 'file_url']
            ]
            print(df_shown.to_markdown(index=False), file=f)
            print('', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=upload_dir,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df), "record")} for azurlane voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/azurlane_voices_index',
    )
