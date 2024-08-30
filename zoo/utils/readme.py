import io

import yaml
from hfutils.operate import get_hf_fs


def get_readme_meta(repo_id: str):
    hf_fs = get_hf_fs()
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
    return meta
