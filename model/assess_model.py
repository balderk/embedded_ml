import os

from evaluation.simple_evaluation import evaluate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    return 100 * relative_absolute_error(y_actual, y_pred)


if __name__ == '__main__':
    model_folder = 'results/'
    model_file_ending = '.h5'
    models_to_assess = ['relu_64_areg_mare_GN0.1_lr0.001', 'relu_256_areg_mare_GN0.1_lr0.001', 'relu_512_areg_mare_GN0.1_lr0.001']

    all_target_keys = sorted(['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)'])
    drop_target = {'NMHC(GT)'}

    target_keys = all_target_keys.copy()
    target_keys.remove(*list(drop_target))

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target),
        drop_outliers=True
    )

    _, test_f_df, _, test_t_df = train_test_split(f, t, test_size=0.3, random_state=1203)
    test_f, test_t = test_f_df.values, test_t_df.values

    for model_to_assess in models_to_assess:
        model = models.load_model(model_folder + model_to_assess + model_file_ending, compile=False)

        # Print the details of the model configuration
        print(model.to_yaml())

        pred = model.predict(test_f)
        pae = percentage_absolute_error(test_t, pred)
        R2 = r2_score(test_t, pred, multioutput='raw_values')

        df_pae = pd.DataFrame(pae, columns=t.columns)
        df_r2 = pd.DataFrame({k: [v] for k, v in zip(t.columns, R2)})

        summary = [['R2 score', *df_r2.values.tolist()[0]], ['MAPE score', *df_pae.mean().to_list()]]
        df_summary = pd.DataFrame(summary, columns=['metric', *df_r2.columns])
        df_summary = df_summary.set_index('metric').T
        print(df_summary)
        df_summary.to_csv(f'summary/{model_to_assess}_summary_performance.csv')
        plt.figure()
        plt.grid(axis='y')
        sns.boxplot(data=df_pae, showfliers=False)
        plt.ylabel('Absolute Percentage Error, lower is better')
        plt.xlabel('Gas component')
        plt.title(f'{model_to_assess}\nOverview of APE based on the different gas components')
        plt.savefig(f'summary/{model_to_assess}_PAE_summary.png')
        plt.show()

        plt.figure()
        plt.grid(axis='y')
        sns.barplot(data=df_r2)
        for i, e in enumerate(R2):
            plt.text(i, e, f'{e:.3f}', horizontalalignment='center', fontweight='bold')
        plt.ylim(min(0.7, R2.min() * 0.9), 1.025)
        plt.ylabel('R2 score, higher is better')
        plt.xlabel('Gas component')
        plt.title(f'{model_to_assess}\nOverview of R2 score based on the different gas components')
        plt.savefig(f'summary/{model_to_assess}_R2_summary.png')
        plt.show()

        base_size = 4
        fig, axs = plt.subplots(1, len(t.columns), figsize=(len(t.columns) * base_size, base_size))
        test_pred = model.predict(test_f)

        for true, pred, ax, target in zip(test_t.T, test_pred.T, axs, t.columns):
            evaluate(true, pred, '', target, ax=ax, set_label=False, include_R2=False, dotted_line=True)

        plt.suptitle(f'{model_to_assess} Scatter plots of predicted vs true values')
        plt.savefig(f'summary/{model_to_assess}_dist_plot.png')

