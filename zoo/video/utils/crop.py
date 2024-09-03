import os
import subprocess
from typing import Union


def crop_video(
        video_path: Union[str, os.PathLike], dst_video_path: Union[str, os.PathLike],
        start_time: float, end_time: float
):
    duration = end_time - start_time
    ffmpeg_cmd = [
        'ffmpeg',
        '-nostdin', '-y',
        '-ss', f'{start_time:.3f}',
        '-i', str(video_path),
        '-t', f'{duration:.3f}',
        '-avoid_negative_ts', '1',
        '-video_track_timescale', '90000',
        '-async', '1',
        str(dst_video_path),
    ]

    with open(os.devnull, 'w') as of_:
        process = subprocess.run(ffmpeg_cmd, stdout=of_, stderr=of_, check=True)
        process.check_returncode()
