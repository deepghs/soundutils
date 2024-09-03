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
        '-i', str(video_path),
        '-ss', f'{start_time:.3f}',
        '-t', f'{duration:.3f}',
        '-avoid_negative_ts', '1',
        '-video_track_timescale', '90000',
        '-async', '1',
        str(dst_video_path),
    ]

    process = subprocess.run(ffmpeg_cmd, check=True)
    process.check_returncode()
