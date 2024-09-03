from unittest.mock import patch

import pytest

from soundutils.video import ffmpeg_cli, ffprobe_cli


# Fixtures
@pytest.fixture
def mock_shutil_which_ffmpeg_found():
    with patch('shutil.which', return_value='/usr/bin/ffmpeg') as mock:
        yield mock


@pytest.fixture
def mock_shutil_which_ffmpeg_not_found():
    with patch('shutil.which', return_value=None) as mock:
        yield mock


@pytest.fixture
def mock_shutil_which_ffprobe_found():
    with patch('shutil.which', return_value='/usr/bin/ffprobe') as mock:
        yield mock


@pytest.fixture
def mock_shutil_which_ffprobe_not_found():
    with patch('shutil.which', return_value=None) as mock:
        yield mock


@pytest.mark.unittest
class TestVideoFFMpeg:
    def test_ffmpeg_cli_found(self, mock_shutil_which_ffmpeg_found):
        assert ffmpeg_cli() == '/usr/bin/ffmpeg'

    def test_ffmpeg_cli_not_found(self, mock_shutil_which_ffmpeg_not_found):
        with pytest.raises(EnvironmentError) as excinfo:
            ffmpeg_cli()
        assert str(excinfo.value) == 'No ffmpeg found in current environment.'

    def test_ffprobe_cli_found(self, mock_shutil_which_ffprobe_found):
        assert ffprobe_cli() == '/usr/bin/ffprobe'

    def test_ffprobe_cli_not_found(self, mock_shutil_which_ffprobe_not_found):
        with pytest.raises(EnvironmentError) as excinfo:
            ffprobe_cli()
        assert str(excinfo.value) == 'No ffprobe found in current environment.'
