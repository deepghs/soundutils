import json
import os
import re
from itertools import islice
from typing import Optional
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import urlsplit, TemporaryDirectory
from hfutils.operate import get_hf_fs, get_hf_client, upload_directory_as_directory
from hfutils.utils import get_requests_session, number_to_tag
from pyquery import PyQuery as pq
from tqdm import tqdm

__root_website__ = 'https://iopwiki.com'

_STAR_PATTERN = re.compile(r'(?P<rarity>\d|EXTRA)star\.png')
_CLASS_PATTERN = re.compile(r'_(?P<class>SMG|MG|RF|HG|AR|SG)_')


def _crawl_index_from_online(session: Optional[requests.Session] = None, maxcnt: Optional[int] = None):
    session = session or get_requests_session()
    resp = session.get(f'{__root_website__}/wiki/T-Doll_Index')
    page = pq(resp.text)

    all_items = list(page('span.gfl-doll-card').items())
    all_items_tqdm = tqdm(all_items)
    for item in all_items_tqdm:
        id_text = item('span.index').text().strip()
        if not id_text:
            continue

        title = item('a').parent('span').attr('title')
        page_url = urljoin(resp.url, item('a').attr('href'))
        sp = urlsplit(page_url)
        assert len(sp.path_segments) == 3 and sp.path_segments[1] == 'wiki', f'Invalid page url - {page_url!r}'
        name_link = sp.path_segments[2]

        rarity = re.findall('doll-rarity-(\d+|EXTRA)', item.attr('class'))[0]
        clazz = re.findall('doll-classification-([a-zA-Z\d]+)', item.attr('class'))[0]
        id_ = int(id_text)

        def _get_name_with_lang(lang: str) -> str:
            return page(f'span[data-server-doll={title!r}][data-server-released={lang!r}]') \
                .attr('data-server-releasename')

        cnname = _get_name_with_lang('CN')
        enname = _get_name_with_lang('EN')
        jpname = _get_name_with_lang('JP')
        all_items_tqdm.set_description(f'{cnname}/{enname}/{jpname}')

        yield {
            'id': id_,
            'name': item('span.name').text().strip(),
            'name_link': name_link,
            'rarity': rarity,
            'class': clazz,

            'cnname': cnname,
            'enname': enname,
            'jpname': jpname,
            'krname': _get_name_with_lang('KR'),
        }


def _get_voices(name: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = session.get(f'https://iopwiki.com/wiki/{quote_plus(name)}/Quotes')
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    page = pq(resp.text)
    retval = []
    for section in page('section.tabber__section').items():
        h2s = list(section.parents('.tabstabber').prev_all('h2').items())
        section_name = h2s[-1]('.mw-headline').text() if h2s else None
        for tab in section('article.tabber__panel').items():
            group_name = tab.attr('data-mw-tabber-title').strip()
            table = tab('table')
            all_rows = list(table('tr').items())
            head_row = all_rows[0]
            assert head_row('th:nth-child(1)').text().strip() == 'Dialogue'
            assert head_row('th:nth-child(3)').text().strip() == 'Japanese'

            last_cnt, last_title = 0, None
            for row in all_rows[1:]:
                if last_cnt > 0:
                    main_box = row('td:nth-child(2)')
                    voice_title = last_title
                else:
                    main_box = row('td:nth-child(3)')
                    voice_title = row('td:nth-child(1)').text().strip()
                    last_cnt = int(row('td:nth-child(1)').attr('rowspan') or 1)
                last_title = voice_title
                last_cnt -= 1

                if not voice_title:
                    logging.warning(f'Voice title is empty for {name!r}, skipped!')
                    continue

                audio_button = main_box('.audio-button')
                if not audio_button:
                    logging.warning(f'Voice {voice_title!r} not playable for {name!r}, skipped.')
                    continue

                main_box.remove('.audio-button')
                voice_text = main_box.text().strip()
                voice_text = re.sub(r'\([\s\S]*\)', '', voice_text).strip()
                if not voice_text:
                    logging.warning(f'Voice {voice_title!r} text of {name!r} is empty, skipped.')
                    continue

                if not audio_button.attr('data-src'):
                    logging.warning(f'No audio file found for voice {voice_title!r} of {name!r}, skipped.')
                    continue
                file_url = urljoin(resp.url, audio_button.attr('data-src'))
                retval.append({
                    'group': f'{section_name} - {group_name}' if section_name else group_name,
                    'voice_title': voice_title,
                    'voice_text': voice_text,
                    'file_url': file_url,
                })

    return retval


class VANotUnique(Exception):
    pass


def _get_info(name: str, session: Optional[requests.Session] = None):
    session = session or get_requests_session()
    resp = session.get(f'https://iopwiki.com/wiki/{quote_plus(name)}')
    resp.raise_for_status()

    page = pq(resp.text)
    table = page('table.profiletable')
    va = None
    for row in table('tr').items():
        htext = row('th:nth-child(1)').text().strip()
        body = row('td:nth-child(2)')
        if 'voice actor' in htext.lower():
            vas = [item.text().strip() for item in body('a').items()]
            if len(vas) != 1:
                logging.warning(f'VA not unique for {name!r} - {vas!r}')
                raise VANotUnique
            else:
                va = vas[0]

    return {
        'voice_actor_name': va,
    }


def sync(repository: str = 'deepghs/girlsfrontline_voices_index', maxcnt: Optional[int] = None):
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
    qs = _crawl_index_from_online(session=session)
    if maxcnt is not None:
        qs = islice(qs, maxcnt)
    d = []
    for chinfo in tqdm(qs):
        try:
            chinfo_extra = _get_info(chinfo['name_link'], session=session)
        except VANotUnique:
            continue
        voices = _get_voices(chinfo['name_link'], session=session)
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
                    'char_krname': chinfo['krname'],
                    'char_rarity': chinfo['rarity'],
                    'char_class': chinfo['class'],
                    'voice_actor_name': chinfo['voice_actor_name'],
                    'voice_id': voice_id,
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
            print('- girlsfrontline', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df))}', file=f)
            print('---', file=f)
            print('', file=f)

            print('# Index Dataset for Girls Front Line Voices', file=f)
            print('', file=f)
            print('This is the middleware index database for the girls front line playable characters.', file=f)
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
            message=f'Sync {plural_word(len(df), "record")} for girlsfrontline voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/girlsfrontline_voices_index',
        maxcnt=None,
    )
