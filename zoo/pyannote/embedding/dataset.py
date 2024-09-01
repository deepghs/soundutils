import os
import shutil
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

repo_id = 'deepghs/arknights_voices_jp'


@lru_cache()
def _df():
    df = pd.read_parquet(hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename='table.parquet'
    )).replace(np.nan, None)
    df = df[(df['time'] >= 3.0) & (df['channels'] == 1) & (df['sample_rate'] == 44100)]
    return df


def _make_nested_dataset(src_dir, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    df = _df()
    df = df[df['filename'].isin(set(os.listdir(src_dir)))]
    print(df)
    quit()

    for row in tqdm(df.to_dict('records')):
        src_file = os.path.join(src_dir, row['filename'])
        dst_file = os.path.join(dst_dir, row['char_id'], row['filename'])
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copyfile(src_file, dst_file)


if __name__ == '__main__':
    _make_nested_dataset(
        src_dir='/data/arknights_jp',
        dst_dir='/data/arknights_jp_nested',
    )
