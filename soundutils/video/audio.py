import logging
import os
import re
import subprocess
from typing import Optional, Union

from hbutils.scale import time_to_duration
from tqdm import tqdm

from .ffmpeg import ffmpeg_cli


def extract_audio_from_video(
        video_path: Union[str, os.PathLike], audio_file: Union[str, os.PathLike],
        sample_rate: Optional[int] = None, silent: bool = False
):
    command = [
        ffmpeg_cli(),
        '-y', '-nostdin',
        '-i', str(video_path),
        *(['-ar', str(sample_rate)] if sample_rate is not None else []),
        '-vn',
        str(audio_file),
    ]
    logging.info(f'Extracting audio from video file with command {command!r} ...')
    with open(os.devnull, 'r') as if_, open(os.devnull, 'r') as of_:
        process = subprocess.Popen(
            command,
            stdin=if_,
            stdout=of_,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

    duration = None
    pg, v = None, 0
    with open(os.devnull, 'w') as ef:
        for line in process.stderr:
            if 'time=' in line and 'speed=' in line:
                if pg is None:
                    pg = tqdm(
                        total=int(round(duration)) if duration is not None else None,
                        desc='Audio Extraction',
                        file=ef if silent else None,
                    )
                time_str, _ = re.findall(r'time=(\d+:\d+:\d+(\.\d+))', line)[0]
                time_val = int(round(time_to_duration(time_str)))
                pg.update(time_val - v)
                v = time_val

            else:
                if pg is None:
                    duration_matching = re.fullmatch(r'^\s*DURATION\s*:\s*(?P<time>\d+:\d+:\d+(\.\d+))\s*$', line)
                    if duration_matching:
                        duration_text = duration_matching.group('time')
                        duration = time_to_duration(duration_text)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)
