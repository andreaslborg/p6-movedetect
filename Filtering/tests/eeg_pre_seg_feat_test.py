import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from eeg_calculate_features import max_func, min_func, slope_func, median_func

'''
Tests for the custom functions used in feature extraction
'''

def test_max_func():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_output = np.array([3, 6])
    assert np.array_equal(max_func(x), expected_output)

def test_min_func():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_output = np.array([1, 4])
    assert np.array_equal(min_func(x), expected_output)

def test_slope_func():
    x = np.array([[1, 2, 3], [2, 4, 6], [-2, -4, -6]])
    expected_output = np.array([1, 2, -2])
    output = slope_func(x)
    np.testing.assert_array_almost_equal(output, expected_output)

def test_median_func():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_output = np.array([2, 5])
    assert np.array_equal(median_func(x), expected_output)

# CMD: pytest RELEARNBackEnd\Processing\Filtering\tests\eeg_pre_seg_feat_test.py