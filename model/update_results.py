import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import tensorflow.keras.backend as kb
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.models as models

from data.get_data import get_feature_targets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

def relative_absolute_error(y_actual, y_pred):
    return np.abs(y_actual - y_pred) / y_actual

def percentage_absolute_error(y_actual, y_pred):
    return 100*relative_absolute_error(y_actual, y_pred)


if __name__ == '__main__':
    df = pd.read_csv('temp_results.csv')

    df.drop_duplicates(subset=['name'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    all_target_keys = sorted(['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)'])
    drop_target = {'NMHC(GT)'}

    target_keys = all_target_keys.copy()
    target_keys.remove(*list(drop_target))

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target),
        drop_outliers=True
    )

    train_f_df, test_f_df, train_t_df, test_t_df = train_test_split(f, t, test_size=0.3, random_state=1203)
    train_f, test_f, train_t, test_t = train_f_df.values, test_f_df.values, train_t_df.values, test_t_df.values
    name_col = ['name']
    pae_dict = dict()
    for model_name, in df[name_col].values:
        print(model_name)
        model = models.load_model(f'{model_name}.h5', compile=False)

        pred = model.predict(test_f)
        pae = percentage_absolute_error(test_t, pred)
        pae_dict[model_name] = pae
        loc = np.where(df[name_col] == model_name)[0]

        for col_name, col in zip(target_keys, pae.T):
            df.loc[loc, f'test {col_name} pae 90'] = np.percentile(col, 90)
            df.loc[loc, f'test {col_name} pae 99'] = np.percentile(col, 99)
            df.loc[loc, f'test {col_name} pae 99.9'] = np.percentile(col, 99.9)
            print(f'{col_name} {np.percentile(col, 90)}, {np.percentile(col, 99)}, {np.percentile(col, 99.9)}')

        df.loc[loc, 'test pae 90 mean'] = np.mean(np.percentile(pae, 90, axis=0))
        df.loc[loc, 'test pae 99 mean'] = np.mean(np.percentile(pae, 99, axis=0))
        df.loc[loc, 'test pae 99.9 mean'] = np.mean(np.percentile(pae, 99.9, axis=0))
    df.to_csv('altered_results.csv', index=False)

    print(df.sort_values('test pae 99.9 mean', ascending=True).head(10))

    plt.boxplot(pae_dict[df.sort_values('test pae 99.9 mean', ascending=True).head(1)['name'].values[0]])
    plt.figure()
    sc = StandardScaler()
    vec = pae_dict[df.sort_values('test pae 99.9 mean', ascending=True).head(1)['name'].values[0]].T
    vec = sc.fit_transform(vec)
    for val in vec:
        print(val)
        sns.distplot(val)

    for k, v in pae_dict.items():
        print(v.argmax(axis=0))
