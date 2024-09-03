import pytest

from soundutils.video import get_video_info
from ..testings import get_testfile


@pytest.fixture()
def example_video_file():
    return get_testfile('assets', 'bangumi_video_crop.mkv')


@pytest.mark.unittest
class TestVideoVideo:
    def test_get_video_info(self, example_video_file):
        video_info = get_video_info(example_video_file)
        assert video_info['width'] == 1920
        assert video_info['height'] == 1080
        assert video_info['duration'] == pytest.approx(6.089)
        assert video_info['fps'] == pytest.approx(23.976023976023978)
        assert video_info['est_nb_frames'] == 145
