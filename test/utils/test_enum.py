import pytest

from soundutils.utils import ExplicitEnum


@pytest.fixture
def sample_enum():
    class Color(ExplicitEnum):
        RED = 'red'
        BLUE = 'blue'
        GREEN = 'green'

    return Color


@pytest.mark.unittest
class TestExplicitEnum:
    def test_valid_enum_value(self, sample_enum):
        assert sample_enum('red') == sample_enum.RED

    def test_invalid_enum_value(self, sample_enum):
        with pytest.raises(ValueError) as exc_info:
            sample_enum('yellow')
        assert "yellow is not a valid Color, please select one of ['red', 'blue', 'green']" in str(exc_info.value)
