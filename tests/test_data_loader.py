import pandas as pd
import pytest

from secomml.data_loader import convert_to_optimal_integer_type


@pytest.mark.parametrize(
	("values", "expected_dtype"),
	[
		([0, 1, 2*128-1], pd.UInt8Dtype()),
		([-128, 0, 127], pd.Int8Dtype()),
		([0, 1, 2*32_768-1], pd.UInt16Dtype()),
		([-32_768, 0, 32_767], pd.Int16Dtype()),
		([0, 1, 2*2_147_483_648-1], pd.UInt32Dtype()),
		([-2_147_483_648, 0, 2_147_483_647], pd.Int32Dtype()),
		([-2_147_483_649, 0, 2_147_483_648], pd.Int64Dtype()),
	],
)
def test_convert_to_optimal_integer_type_uses_smallest_signed_dtype(
	values, expected_dtype
):
	series = pd.DataFrame({'x':values}, dtype="int64")

	result = convert_to_optimal_integer_type(series)

	assert result['x'].dtype == expected_dtype
	assert result.to_numpy().flatten().tolist() == values
