from ditk import logging

from .db import sync

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        src_repo='deepghs/azurlane_voices_index',
        dst_repo='deepghs/azurlane_voices_jp',
    )
