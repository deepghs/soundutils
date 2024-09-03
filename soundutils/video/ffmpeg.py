import shutil


def ffmpeg_cli():
    exec = shutil.which('ffmpeg')
    if not exec:
        raise EnvironmentError('No ffmpeg found in current environment.')
    else:
        return exec


def ffprobe_cli():
    exec = shutil.which('ffprobe')
    if not exec:
        raise EnvironmentError('No ffprobe found in current environment.')
    else:
        return exec
