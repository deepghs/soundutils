import json
import os
import subprocess
from typing import Union, Optional

from .ffmpeg import ffprobe_cli


def get_video_full_info(video_path: Union[str, os.PathLike]) -> dict:
    command = [
        ffprobe_cli(),
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    result.check_returncode()
    return json.loads(result.stdout)


def get_video_info(video_path: Union[str, os.PathLike]) -> Optional[dict]:
    video_info = get_video_full_info(video_path)
    streams = video_info.get('streams', [])
    video_stream = next((stream for stream in streams if stream['codec_type'] == 'video'), None)

    if video_stream:
        width = video_stream.get('width')
        height = video_stream.get('height')
        duration = float(video_info['format'].get('duration', 0))

        avg_frame_rate = video_stream.get('avg_frame_rate')
        if avg_frame_rate:
            fps = eval(avg_frame_rate)
            estimated_frame_count = int(duration * fps)
        else:
            fps = None
            estimated_frame_count = None

        return {
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'est_nb_frames': estimated_frame_count,
            'full': video_info
        }
    else:
        return None
