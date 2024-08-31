import os.path

import pytest

from soundutils.langid import silero_langid
from test.testings import get_testfile

_MAP = {
    'en': ['en, English'],
    'zh': ['zh, Chinese', 'zh-CN, Chinese'],
    'jp': ['ja, Japanese'],
    'kr': ['ko, Korean'],
}


@pytest.mark.unittest
class TestLangidSilero:
    @pytest.mark.parametrize(['file', 'lang'], [
        (os.path.join(lang, f'{lang}_{item}.wav'), lang)
        for item in ['short', 'medium', 'long']
        for lang in ['en', 'jp', 'zh', 'kr']
    ])
    def test_silero_langid(self, file, lang):
        label, score = silero_langid(get_testfile('assets', 'langs', file))
        assert label in _MAP[lang]
        assert isinstance(score, float)
