import subprocess
from unittest.mock import patch

import pytest
from hbutils.testing import isolated_directory

from soundutils.similarity import sound_mfcc_similarity
from soundutils.video import extract_audio_from_video
from ..testings import get_testfile


@pytest.fixture()
def example_video_file():
    return get_testfile('assets', 'bangumi_video_crop.mkv')


@pytest.fixture
def mock_subprocess_popen():
    class MockProcess:
        def __init__(self):
            self.returncode = 1
            self.args = 'ffmpeg -y -nostdin -i input.mp4 output.mp3'
            self.stderr = iter([
                "DURATION: 00:01:00.00",
                "time=00:00:30.00 speed=2x"
            ])

        def wait(self):
            return self.returncode

    return MockProcess


@pytest.mark.unittest
class TestVideoAudio:
    def test_extract_audio_from_video(self, example_video_file):
        with isolated_directory():
            extract_audio_from_video(example_video_file, 'audio.wav')
            assert sound_mfcc_similarity(
                'audio.wav',
                get_testfile('assets', 'bangumi_video_crop_audio.wav')
            ) >= 0.95

    def test_extract_audio_from_video_error(self, mock_subprocess_popen):
        with patch('subprocess.Popen', return_value=mock_subprocess_popen()):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                extract_audio_from_video('input.mp4', 'output.mp3')
            assert exc_info.value.returncode == 1
