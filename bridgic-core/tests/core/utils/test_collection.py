"""
Test cases for collection utility functions.
"""
import pytest
from bridgic.core.utils.collection import filter_dict, unique_list_in_order, deep_hash


def test_filter_dict_basic_none_filtering():
    """Test basic filtering of None values."""
    data = {"a": 1, "b": None, "c": 3, "d": "test"}
    result = filter_dict(data)
    expected = {"a": 1, "c": 3, "d": "test"}
    assert result == expected


def test_filter_dict_exclude_none_false():
    """Test when exclude_none is False."""
    data = {"a": 1, "b": None, "c": 3}
    result = filter_dict(data, exclude_none=False)
    expected = {"a": 1, "b": None, "c": 3}
    assert result == expected


def test_filter_dict_custom_exclude_values():
    """Test filtering with custom exclude values."""
    data = {"a": 1, "b": "exclude_me", "c": 3, "d": "keep_me"}
    result = filter_dict(data, exclude_values=("exclude_me",))
    expected = {"a": 1, "c": 3, "d": "keep_me"}
    assert result == expected


def test_filter_dict_multiple_exclude_values():
    """Test filtering with multiple exclude values."""
    data = {"a": 1, "b": "exclude1", "c": 3, "d": "exclude2", "e": "keep"}
    result = filter_dict(data, exclude_values=("exclude1", "exclude2"))
    expected = {"a": 1, "c": 3, "e": "keep"}
    assert result == expected


def test_filter_dict_combine_none_and_custom_exclude():
    """Test combining None filtering with custom exclude values."""
    data = {"a": 1, "b": None, "c": "exclude_me", "d": 3}
    result = filter_dict(data, exclude_values=("exclude_me",))
    expected = {"a": 1, "d": 3}
    assert result == expected


def test_filter_dict_exclude_none_false_with_custom_exclude():
    """Test with exclude_none=False and custom exclude values."""
    data = {"a": 1, "b": None, "c": "exclude_me", "d": 3}
    result = filter_dict(data, exclude_none=False, exclude_values=("exclude_me",))
    expected = {"a": 1, "b": None, "d": 3}
    assert result == expected


def test_filter_dict_empty_dict():
    """Test with empty dictionary."""
    data = {}
    result = filter_dict(data)
    expected = {}
    assert result == expected


def test_filter_dict_all_none_values():
    """Test with all None values."""
    data = {"a": None, "b": None, "c": None}
    result = filter_dict(data)
    expected = {}
    assert result == expected


def test_filter_dict_all_excluded_values():
    """Test with all values being excluded."""
    data = {"a": "exclude1", "b": "exclude2", "c": None}
    result = filter_dict(data, exclude_values=("exclude1", "exclude2"))
    expected = {}
    assert result == expected


def test_filter_dict_no_filtering_needed():
    """Test when no filtering is needed."""
    data = {"a": 1, "b": 2, "c": 3}
    result = filter_dict(data)
    expected = {"a": 1, "b": 2, "c": 3}
    assert result == expected


def test_filter_dict_numeric_zero_and_false():
    """Test that 0 and False are not filtered out."""
    data = {"a": 0, "b": False, "c": None, "d": 1}
    result = filter_dict(data)
    expected = {"a": 0, "b": False, "d": 1}
    assert result == expected


def test_filter_dict_empty_string():
    """Test that empty string is not filtered out."""
    data = {"a": "", "b": None, "c": "text"}
    result = filter_dict(data)
    expected = {"a": "", "c": "text"}
    assert result == expected


def test_filter_dict_complex_data_types():
    """Test with complex data types."""
    data = {
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "none_val": None,
        "tuple": (1, 2, 3)
    }
    result = filter_dict(data)
    expected = {
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "tuple": (1, 2, 3)
    }
    assert result == expected


def test_filter_dict_object_identity_exclude():
    """Test that object identity is used for exclude_values."""
    exclude_obj = object()
    data = {"a": 1, "b": exclude_obj, "c": 3}
    result = filter_dict(data, exclude_values=(exclude_obj,))
    expected = {"a": 1, "c": 3}
    assert result == expected


def test_filter_dict_same_value_different_objects():
    """Test that different objects with same value are not excluded."""
    data = {"a": 1, "b": object(), "c": 3}
    exclude_obj = object()  # Different object
    result = filter_dict(data, exclude_values=(exclude_obj,))
    expected = {"a": 1, "b": data["b"], "c": 3}
    assert result == expected


def test_filter_dict_original_dict_unchanged():
    """Test that original dictionary is not modified."""
    data = {"a": 1, "b": None, "c": 3}
    original_data = data.copy()
    filter_dict(data)
    assert data == original_data


def test_filter_dict_return_new_dict():
    """Test that a new dictionary is returned."""
    data = {"a": 1, "b": None, "c": 3}
    result = filter_dict(data)
    assert result is not data  # Different object
    assert isinstance(result, dict)


def test_unique_list_in_order_basic_functionality():
    """Test basic deduplication while preserving order."""
    data = [1, 2, 3, 2, 4, 1, 5]
    result = unique_list_in_order(data)
    expected = [1, 2, 3, 4, 5]
    assert result == expected


def test_unique_list_in_order_empty_list():
    """Test with empty list."""
    data = []
    result = unique_list_in_order(data)
    expected = []
    assert result == expected


def test_unique_list_in_order_no_duplicates():
    """Test with no duplicates."""
    data = [1, 2, 3, 4, 5]
    result = unique_list_in_order(data)
    expected = [1, 2, 3, 4, 5]
    assert result == expected


def test_unique_list_in_order_all_duplicates():
    """Test with all same elements."""
    data = [1, 1, 1, 1]
    result = unique_list_in_order(data)
    expected = [1]
    assert result == expected


def test_unique_list_in_order_mixed_types():
    """Test with mixed data types."""
    data = [1, "a", 2, "b", 1, "a", 3]
    result = unique_list_in_order(data)
    expected = [1, "a", 2, "b", 3]
    assert result == expected


def test_deep_hash_basic_types():
    """Test basic hashable types."""
    assert deep_hash("string") == hash("string")
    assert deep_hash(42) == hash(42)
    assert deep_hash(3.14) == hash(3.14)
    assert deep_hash(True) == hash(True)
    assert deep_hash(None) == hash(None)


def test_deep_hash_list_and_tuple():
    """Test with lists and tuples."""
    data_list = [1, 2, 3]
    data_tuple = (1, 2, 3)
    assert deep_hash(data_list) == deep_hash(data_tuple)
    assert deep_hash(data_list) == hash((1, 2, 3))


def test_deep_hash_nested_structures():
    """Test with nested structures."""
    data = {"a": [1, 2], "b": {"c": 3}}
    result = deep_hash(data)
    assert isinstance(result, int)
    assert result == deep_hash(data)  # Consistent hash


def test_deep_hash_set():
    """Test with sets."""
    data = {1, 2, 3}
    result = deep_hash(data)
    assert isinstance(result, int)
    assert result == deep_hash({3, 2, 1})  # Order independent


def test_deep_hash_unhashable_type_error():
    """Test that unhashable types raise TypeError."""
    class UnhashableClass:
        def __hash__(self):
            raise TypeError("Unhashable type")
    
    with pytest.raises(TypeError, match="Unhashable type"):
        deep_hash(UnhashableClass())


def test_deep_hash_consistent_hashing():
    """Test that same data produces same hash."""
    data1 = {"a": [1, 2], "b": {"c": 3}}
    data2 = {"b": {"c": 3}, "a": [1, 2]}  # Different order
    assert deep_hash(data1) == deep_hash(data2)
