import os
import re
import shlex
import subprocess
from fractions import Fraction

from PIL import Image
from tqdm import tqdm

from soundutils.video import get_video_info, ffmpeg_cli


def extract_frames(video_path, keyframes: bool = True, per_seconds: float = None, per_frames: int = None):
    video_info = get_video_info(video_path)
    width, height = video_info['width'], video_info['height']
    duration = int(round(video_info['duration'] * 1000))

    if keyframes:
        vf = 'select=eq(pict_type\,I),showinfo'
    else:
        if per_seconds is not None:
            frac = Fraction(1 / per_seconds).limit_denominator(3000)
            vf = f'fps={frac},showinfo'
        elif per_frames:
            vf = f'select=not(mod(n\,{per_frames})),showinfo'
        else:
            raise ValueError('Unable to determine how to crop.')

    command = [
        ffmpeg_cli(),
        '-i', video_path,
        '-vf', vf,
        '-vsync', '0',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    print(shlex.join(command))

    frame_pattern = re.compile(r'n:\s*(?P<order>\d+).*?\s*pts:\s*(?P<pts>\d+).*?\s*pts_time:\s*(?P<pts_time>[\d.]+)')
    frame_size = width * height * 3

    v = 0
    with open(os.devnull, 'r') as if_, tqdm(total=duration, desc='Extract Frames', unit='ms') as pg:
        process = subprocess.Popen(command, stdin=if_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for line in process.stderr:
                match = frame_pattern.search(line.decode())
                if match:
                    order_id = int(match.group('order'))
                    time_secs = float(match.group('pts_time'))

                    raw_frame = process.stdout.read(frame_size)
                    image = Image.frombytes('RGB', (width, height), raw_frame)
                    new_v = int(round(time_secs * 1000))
                    pg.update(new_v - v)
                    v = new_v
                    yield order_id, time_secs, image
        finally:
            exitcode = process.poll()
            if exitcode is None:
                process.kill()
                process.wait()
            else:
                if exitcode != 0:
                    raise subprocess.CalledProcessError(
                        returncode=exitcode,
                        cmd=command,
                    )
