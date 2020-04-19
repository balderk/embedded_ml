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
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import seaborn as sns

def relative_absolute_error(y_actual, y_pred):
    return np.abs(y_actual - y_pred) / y_actual

def percentage_absolute_error(y_actual, y_pred):
    return 100*relative_absolute_error(y_actual, y_pred)


if __name__ == '__main__':
    model_folder = 'results/'
    model_file_ending = '.h5'
    model_to_assess = 'relu_256_kareg_mare_GN0.05_lr0.001'  # 'relu_512_kareg_mare_GN0.1_lr0.001'

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

    model = models.load_model(model_folder + model_to_assess + model_file_ending, compile=False)

    # Print the details of the model configuration
    print(model.to_yaml())

    pred = model.predict(test_f)
    pae = percentage_absolute_error(test_t, pred)
    R2 = r2_score(test_t, pred, multioutput='raw_values')

    df_pae = pd.DataFrame(pae, columns=t.columns)
    df_r2 = pd.DataFrame({k: [v] for k, v in zip(t.columns, R2)})

    plt.figure()
    sns.boxplot(data=df_pae, showfliers=False)
    plt.ylabel('Percentage absolute error')
    plt.xlabel('Gas component')
    plt.title('Overview of percentage absolute error based on the different gas components')
    plt.grid()
    plt.show()

    plt.figure()
    sns.barplot(data=df_r2)
    plt.ylabel('R2 score')
    plt.xlabel('Gas component')
    plt.title('Overview of R2 score based on the different gas components')
    plt.show()

    #sc = StandardScaler()
    #vec = pae_dict[df.sort_values('test pae 99.9 mean', ascending=True).head(1)['name'].values[0]].T
    #vec = sc.fit_transform(vec)
    #for val in vec:
    #    print(val)
    #    sns.distplot(val)

