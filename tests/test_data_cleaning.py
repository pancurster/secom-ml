import numpy as np
import pandas as pd
import pytest

from secomml.data_cleaning import drop_zero_percentage_columns


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
