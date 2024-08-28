from ditk import logging

from .db import sync

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        src_repo='deepghs/girlsfrontline_voices_index',
        dst_repo='deepghs/girlsfrontline_voices_jp',
    )
