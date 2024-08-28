import json
import os
import re
from typing import List, Optional
from urllib.parse import quote, urljoin

import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import get_hf_client, get_hf_fs, upload_directory_as_directory
from hfutils.utils import get_requests_session, number_to_tag
from pyquery import PyQuery as pq
from tqdm.auto import tqdm

SERVANT_ALT_PATTERN = re.compile(r'Servant (?P<id>\d+)\.[a-zA-Z\d]+')
PAGE_REL_PATTERN = re.compile(r'var data_list\s*=\"(?P<ids>[\d,\s]*)\"')


class FateGrandOrderIndexer:
    __game_name__ = 'fgo'
    __root_website__ = 'https://fgo.wiki/'

    def _get_alias_of_op(self, op, session: requests.Session, names: List[str]) -> List[str]:
        response = session.get(
            f'{self.__root_website__}/api.php?action=query&prop=redirects&titles={quote(op)}&format=json',
        )
        response.raise_for_status()

        alias_names = []
        pages = response.json()['query']['pages']
        for _, data in pages.items():
            for item in (data.get('redirects', None) or []):
                if item['title'] not in names:
                    alias_names.append(item['title'])

        return alias_names

    def _get_similar_lists(self, current_id: int, sim_table: pq, session: requests.Session = None) -> List[int]:
        session = session or get_requests_session()
        content_box = sim_table('td')
        ids = []
        if content_box('td a img'):
            for image in content_box('td a img').items():
                sid = int(SERVANT_ALT_PATTERN.fullmatch(image.attr('alt')).group('id').lstrip('0'))
                ids.append(sid)

        elif content_box('td a'):
            sim_page_url = f"{self.__root_website__}/{content_box('td a').attr('href')}"
            resp = session.get(sim_page_url)

            for reltext in PAGE_REL_PATTERN.findall(resp.content.decode()):
                for id_ in [int(item.strip()) for item in reltext.split(',')]:
                    ids.append(id_)

        else:
            raise ValueError(f'Unknown similar table content:{os.linesep}{content_box}.')  # pragma: no cover

        return sorted(set([id_ for id_ in ids if current_id != id_]))

    def _crawl_index_from_online(self, session: requests.Session, maxcnt: Optional[int] = None, **kwargs):
        response = session.get(f'{self.__root_website__}/w/SVT')
        (raw_text, *_), *_ = re.findall(r'override_data\s*=\s*(?P<str>"(\\"|[^"])+")', response.text)
        raw_text: str = eval(raw_text)

        fulllist, curobj = [], {}
        for line in raw_text.splitlines(keepends=False):
            line = line.strip()
            if line:
                name, content = line.split('=', maxsplit=1)
                curobj[name] = content
            else:
                fulllist.append(curobj)
                curobj = {}

        if curobj:
            fulllist.append(curobj)

        _id_to_name = {int(item['id']): item['name_cn'] for item in fulllist}
        fulllist = tqdm(fulllist[::-1], leave=True)
        retval = []
        for item in fulllist:
            id_ = int(item['id'])
            cnname = item['name_cn'].replace('・', '·')
            jpname = item['name_jp']
            enname = item['name_en']
            fulllist.set_description(cnname)

            alias = [name.strip() for name in item['name_other'].split('&') if name.strip()]
            get_method = item['method']

            resp = session.get(f'{self.__root_website__}/w/{quote(item["name_link"])}')
            page = pq(resp.text)
            main_table, *_other_tables = page('table.wikitable.nomobile').items()

            row1 = main_table('tr:nth-child(1)')
            row2 = main_table('tr:nth-child(2)')
            row3 = main_table('tr:nth-child(3)')
            row5 = main_table('tr:nth-child(5)')
            row7 = main_table('tr:nth-child(7)')

            CN_ALIAS_PATTERN = re.compile('^(?P<cnalias>[^（]+)（(?P<cnname>[^）]+)）$')
            if CN_ALIAS_PATTERN.fullmatch(row1('th:nth-child(1)').text()) and row1('th:nth-child(1) span'):
                matching = CN_ALIAS_PATTERN.fullmatch(row1('th:nth-child(1)').text())
                cn_alias = matching.group('cnalias')
                if cn_alias not in alias:
                    alias.append(cn_alias)
                all_cnnames = [matching.group('cnname')]
                all_jpnames = [row2('td:nth-child(1)').text()]
                all_ennames = [row3('td:nth-child(1)').text()]
            else:
                s_r1 = row1('th:nth-child(1)').text()
                s_r2 = row2('td:nth-child(1)').text()
                s_r3 = row3('td:nth-child(1)').text()
                if '/' in s_r1 and '/' in s_r2 and '/' in s_r3:
                    all_cnnames = s_r1.split('/')
                    all_jpnames = s_r2.split('/')
                    all_ennames = s_r3.split('/')
                    assert len(all_cnnames) == len(all_jpnames) == len(all_ennames)
                else:
                    all_cnnames = [s_r1]
                    all_jpnames = [s_r2]
                    all_ennames = [s_r3]

            if cnname not in all_cnnames:
                all_cnnames.append(cnname)
            if jpname not in all_jpnames:
                all_jpnames.append(jpname)
            if enname not in all_ennames:
                all_ennames.append(enname)

            accessible = not (row1('th > span').text().strip() == '无法召唤')
            if not row3('th > img'):
                logging.info(f'No rarity found for #{id_}, ({all_cnnames!r}).')
                rarity = None
            else:
                rarity = int(re.findall(r'\d+', row3('th > img').attr('alt'))[0])
            clazz = row7('td:nth-child(1)').text().strip()
            gender = row7('td:nth-child(2)').text().strip()

            va_ele = row5('td:nth-child(2)')
            if id_ == 1:
                *_, aitem = va_ele('a').items()
                va = aitem.text().strip()
            else:
                if len(list(va_ele('a').items())) != 1:
                    va_names = [aitem.text().strip() for aitem in va_ele('a').items()]
                    logging.warning(f'Multiple va found for #{id_}, ({all_cnnames!r}), skipped - {va_names!r}.')
                    continue
                aitem, = va_ele('a').items()
                va = aitem.text().strip()

            alias.extend(
                self._get_alias_of_op(item["name_link"], session, [*all_cnnames, *all_ennames, *all_jpnames, *alias]))
            retval.append({
                'id': id_,
                'jpname': all_jpnames[0] if all_jpnames else None,
                'cnname': all_cnnames[0] if all_cnnames else None,
                'enname': all_ennames[0] if all_ennames else None,
                'jpnames': all_jpnames,
                'cnnames': all_cnnames,
                'ennames': all_ennames,
                'name': item['name_link'],
                'alias': alias,
                'accessible': accessible,
                'rarity': rarity,
                'method': get_method,
                'class': clazz,
                'gender': gender,
                'voice_actor_name': va,
                'voices': self._get_voices(item['name_link'], session=session),
            })
            if maxcnt is not None and len(retval) >= maxcnt:
                break

        return retval

    def _get_voices(self, name, session: Optional[requests.Session] = None):
        session = session or get_requests_session()
        resp = session.get(f'{self.__root_website__}/w/{quote(name)}/语音')
        page = pq(resp.text)

        retval = []
        for i, table in enumerate(page('table.wikitable.nomobile').items()):
            btitle = list(table.prev_all('h2').items())[-1].text()
            first_row = list(table('tr').items())[0]
            if len(list(first_row('th').items())) == 1 and \
                    len(list(first_row('td').items())) == 2:
                for row in table('tr').items():
                    voice_title = row('th').text().strip()
                    voice_text = row('td:nth-child(2) p[lang=ja]').text().strip()
                    if not re.sub(r'\([\s\S]*\)', '', voice_text):
                        logging.warning(f'No voice text found for {name!r}/{voice_title!r}, skipped.')
                        continue

                    if not row('td:nth-child(3) a[download]'):
                        logging.warning(f'No voice downloadable url found for {name!r}/{voice_title!r}, skipped.')
                        continue
                    else:
                        file_url = urljoin(resp.url, row('td:nth-child(3) a[download]').attr('href'))
                        retval.append({
                            'group': btitle,
                            'voice_title': voice_title,
                            'voice_text': voice_text,
                            'file_url': file_url,
                        })
            else:
                logging.warning(f'Invalid table for {name!r} - {btitle!r}, skipped.')
                continue
        return retval

    def get_index(self, maxcnt: Optional[int] = None):
        return self._crawl_index_from_online(
            session=get_requests_session(),
            maxcnt=maxcnt,
        )


def sync(repository: str = f'deepghs/fgo_voices_index', maxcnt: Optional[int] = None):
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

    idx = FateGrandOrderIndexer()
    d: list = idx.get_index(maxcnt=maxcnt)

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
                    'char_zhname': chinfo['enname'],
                    'char_jpnames': chinfo['jpnames'],
                    'char_cnnames': chinfo['cnnames'],
                    'char_zhnames': chinfo['ennames'],
                    'char_rarity': chinfo['rarity'],
                    'char_gender': chinfo['gender'],
                    'char_class': chinfo['class'],
                    'char_accessible': chinfo['accessible'],
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
            print('- fgo', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df))}', file=f)
            print('---', file=f)
            print('', file=f)

            print('# Index Dataset for FGO Voices', file=f)
            print('', file=f)
            print('This is the middleware index database for the FGO playable characters.', file=f)
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
            message=f'Sync {plural_word(len(df), "record")} for fgo voices'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/fgo_voices_index',
        maxcnt=5,
    )
