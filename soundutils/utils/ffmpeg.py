import shutil


def ffmpeg_cli():
    exec = shutil.which('ffmpeg')
    if not exec:
        raise EnvironmentError('No ffmpeg found in current environment.')
    else:
        return exec
