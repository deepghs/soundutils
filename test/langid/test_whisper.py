import os

import pytest

from soundutils.langid import whisper_langid
from test.testings import get_testfile


@pytest.mark.unittest
class TestLangidWhisper:
    @pytest.mark.parametrize(['lang', 'file'], [
        ('Mandarin Chinese', os.path.join('zh', 'zh_long.wav')),
        ('Japanese', os.path.join('jp', 'jp_long.wav')),
        ('Korean', os.path.join('kr', 'kr_long.wav')),
        ('English', os.path.join('en', 'en_long.wav')),
    ])
    def test_whisper_langid(self, lang, file):
        alang, ascore = whisper_langid(get_testfile('assets', 'langs', file))
        assert alang == lang
