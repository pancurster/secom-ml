from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold

def secom_simple_imputer(df, strategy='mean'):
    return SimpleImputer(strategy=strategy).set_output(transform='pandas').fit_transform(df)

def secom_standard_scaler(df):
    return StandardScaler().set_output(transform='pandas').fit_transform(df)

def secom_minmax_scaler(df, feature_range=(0,1)):
    return MinMaxScaler(feature_range=feature_range).set_output(transform='pandas').fit_transform(df)

def secom_normalizer_scaler(df, norm='l2'):
    return Normalizer(norm=norm).set_output(transform='pandas').fit_transform(df)

def secom_drop_zero_variance_features(df):
    return VarianceThreshold(threshold=0.0).set_output(transform='pandas').fit_transform(df)

def secom_drop_ths_variance_features(df, threshold=0.0):
    return VarianceThreshold(threshold=threshold).set_output(transform='pandas').fit_transform(df)