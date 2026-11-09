import numpy as np
import pandas as pd
import pytest

from secomml.data_cleaning import MaxNanFeatureSelector, drop_zero_percentage_columns


def test_nan_feature_selector_drops_features_with_more_than_90_percent_missing():
    frame = pd.DataFrame(
        {
            "keep": [1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan, 5.0, np.nan],
            "drop": [np.nan] * 9 + [1.0],
        }
    )

    selector = MaxNanFeatureSelector()
    result = selector.fit_transform(frame)

    assert selector.get_feature_names_out().tolist() == ["keep"]
    assert result.shape == (10, 1)
    assert np.isnan(result).any()


def test_nan_feature_selector_supports_numpy_arrays():
    values = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])

    result = MaxNanFeatureSelector().fit_transform(values)

    np.testing.assert_array_equal(result, values[:, [0]])


def test_drops_columns_with_only_zero_values():
	frame = pd.DataFrame(
		{
			"all_zero": np.zeros(100),
			"98_zeros": np.concat([np.zeros(98), np.ones(2)]),
			"97_zeros": np.concat([np.zeros(97), np.ones(3)]),
			"nonzero": np.random.chisquare(1, 100),
			'90z_10na': np.concat([np.zeros(90), np.ones(10)*np.nan]),
		}
	)

	result = drop_zero_percentage_columns(frame, ths=0.98, na_policy='as_zero')

	assert list(result.columns) == ["97_zeros", "nonzero"]
	pd.testing.assert_frame_equal(result, frame[["97_zeros", "nonzero"]])
