import os
import numpy as np
import pandas as pd

DATA_FILENAME = 'AirQuality.xlsx'
POSSIBLE_FOLDERS = ['master_embedded_ml']  # ['embedded_ml']

_ABS_PATH = os.path.abspath('.')
print(_ABS_PATH)
for folder in _ABS_PATH.split(os.path.sep)[::-1]:
    if folder in POSSIBLE_FOLDERS:
        DATA_FOLDER_NAME = os.path.join(_ABS_PATH[:_ABS_PATH.index(folder)], folder, 'data')
        break


def get_filename():
    return os.path.abspath(os.path.join(DATA_FOLDER_NAME, DATA_FILENAME))


def get_data() -> pd.DataFrame:
    # target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    df = pd.read_excel(get_filename())

    for key in df.columns:
        df[key].replace(-200, np.NaN, inplace=True)

    return df


def get_n_data(n=1000, random_seed=123) -> pd.DataFrame:
    return get_data().sample(n, random_state=random_seed)


def get_feature_targets(dropna=False, feature_keys=None, target_keys=None, drop_outliers=False) -> (pd.DataFrame, pd.DataFrame):
    if feature_keys is None:
        feature_keys = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    if target_keys is None:
        target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

    data = get_data()[feature_keys + target_keys]
    if dropna:
        data = data.dropna()
        data.reset_index(drop=True, inplace=True)
        if drop_outliers:
            data.drop([4258, 4267, 4288, 4289, 4356, 4742], inplace=True)  # hardcoded, bad practice
            data.reset_index(drop=True, inplace=True)
            data.drop([110, 154, 155, 3700, 3891, 4154, 4257, 4265, 4266, 4286, 4656, 4658, 4894, 5313, 5314],
                      inplace=True)  # hardcoded, bad practice
            data.reset_index(drop=True, inplace=True)

    features = data[sorted(feature_keys)]
    targets = data[sorted(target_keys)]

    return features, targets


if __name__ == '__main__':
    f, t = get_feature_targets(dropna=True, )
    print(f.shape)
