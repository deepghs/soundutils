import pytest

from soundutils.utils import vreplace


@pytest.fixture
def simple_mapping():
    return {'a': 'apple', 'b': 'banana'}


@pytest.fixture
def complex_mapping():
    return {1: 'one', 2: 'two', 3: 'three'}


@pytest.fixture
def nested_data_structure():
    return [1, {'a': 2, 'b': [3, 4]}]


@pytest.mark.unittest
class TestVReplace:
    def test_basic_replacement(self, simple_mapping):
        assert vreplace('a', simple_mapping) == 'apple'
        assert vreplace('b', simple_mapping) == 'banana'

    def test_no_replacement(self, simple_mapping):
        assert vreplace('c', simple_mapping) == 'c'

    def test_list_replacement(self, simple_mapping):
        assert vreplace(['a', 'b', 'c'], simple_mapping) == ['apple', 'banana', 'c']

    def test_tuple_replacement(self, simple_mapping):
        assert vreplace(('a', 'b', 'c'), simple_mapping) == ('apple', 'banana', 'c')

    def test_dict_replacement_1(self, complex_mapping):
        input_dict = {'a': 1, 'b': 2}
        expected_dict = {'a': 'one', 'b': 'two'}
        assert vreplace(input_dict, complex_mapping) == expected_dict

    def test_dict_replacement_2(self, complex_mapping):
        input_dict = {1: 'a', 2: 'b'}
        expected_dict = {1: 'a', 2: 'b'}
        assert vreplace(input_dict, complex_mapping) == expected_dict

    def test_nested_structure_replacement(self, complex_mapping, nested_data_structure):
        expected_structure = ['one', {'a': 'two', 'b': ['three', 4]}]
        assert vreplace(nested_data_structure, complex_mapping) == expected_structure

    def test_unhashable_type(self):
        unhashable = [1, 2, {3}]
        assert vreplace(unhashable, {1: 'one'}) == ['one', 2, {3}]
