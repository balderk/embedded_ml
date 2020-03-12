import os
import tensorflow as tf

from data.get_data import get_feature_targets
from evaluation.simple_evaluation import evaluate

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow.keras.backend as kb

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.utils import model_to_dot, plot_model


def mean_relative_mae(y_actual, y_pred) -> tf.Tensor:
    return kb.mean(kb.abs(y_actual - y_pred) / y_actual)


def max_relative_mae(y_actual, y_pred) -> tf.Tensor:
    return kb.max(kb.abs(y_actual - y_pred) / y_actual)


def np_mean_relative_mae(y_actual, y_pred):
    return np.mean(np.abs(y_actual - y_pred) / y_actual)


def np_max_relative_mae(y_actual, y_pred):
    return np.max(np.abs(y_actual - y_pred) / y_actual)


def train_model(config: dict) -> dict:
    name = config.get('name', None)
    model = Sequential(config.get('model', None))
    ret_dict = {'name': name}
    opt = config.get('optimizer', optimizers.Adam(lr=1e-3))

    model.compile(
        optimizer=opt,
        loss='mae',
        metrics=['mean_absolute_error', max_relative_mae]
        # loss='mean_absolute_error',
        # metrics=['mean_absolute_error']
    )
    plot_model(model, to_file=f'{name}_model.png', show_shapes=True, expand_nested=True)
    thing = model_to_dot(model, show_shapes=True, expand_nested=True).create(prog='dot',
                                                                             format='svg')  # .create_svg(f'{name}.svg')
    with open(f'{name}.svg', 'wb') as fil:
        fil.write(thing)
    fig = plt.figure(1)
    fig.clf()
    fig, axs = plt.subplots(1, 1, num=1)
    mae = []
    mae_val = []

    line_mae = axs.plot(mae, range(len(mae)), 'b-', label='mae')[0]

    line_mae_val = axs.plot(mae_val, range(len(mae_val)), 'r-', label='mae_val')[0]

    axs.grid()
    axs.set_title(f'{name} loss')
    axs.legend()
    plt.pause(1e-3)
    total_epoch = config.get('total_epoch', 100000)
    val_freq = config.get('val_freq', 1000)
    epoch_chunk_size = config.get('epoch_chunk_size', 10000)
    patience = config.get('patience', 5)
    chunks = np.arange(0, total_epoch, epoch_chunk_size, dtype='int')
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
    for ini_epoch, epoch in zip(chunks[:-1], chunks[1:]):
        hist = model.fit(
            train_f,
            train_t,
            initial_epoch=ini_epoch,
            epochs=epoch,
            batch_size=len(train_t),
            validation_data=(test_f, test_t),
            validation_freq=val_freq,
            callbacks=[early_stop],
            verbose=False,
        )

        mae.extend(hist.history['max_relative_mae'])

        for val in hist.history['val_max_relative_mae']:
            mae_val.extend([val for _ in range(val_freq)])

        line_mae.set_data(range(len(mae)), mae)
        line_mae_val.set_data(range(len(mae_val)), mae_val)
        axs.set_xlim(0, len(mae))
        center_num = epoch_chunk_size
        tmp_min = min([min(mae[-center_num:]), min(mae_val[-center_num:])])
        tmp_max = max([max(mae[-center_num:]), max(mae_val[-center_num:])])
        axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

        plt.pause(1e-3)
        if len(hist.history['loss']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(mae), min(mae_val)])
    tmp_max = max([max(mae), max(mae_val)])
    axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

    plt.pause(1e-3)
    plt.savefig(f'{name}_loss_stage_1.png')

    model.compile(
        optimizer=opt,
        loss="mae",  # mean_relative_mae,
        metrics=['mean_absolute_error', max_relative_mae]
        # loss='mean_absolute_error',
        # metrics=['mean_absolute_error']
    )

    for ini_epoch, epoch in zip(chunks[:-1], chunks[1:]):
        hist = model.fit(
            train_f,
            train_t,
            initial_epoch=ini_epoch,
            epochs=epoch,
            batch_size=len(train_t),
            validation_data=(test_f, test_t),
            validation_freq=val_freq,
            callbacks=[early_stop],
            verbose=False,
        )

        mae.extend(hist.history['max_relative_mae'])

        for val in hist.history['val_max_relative_mae']:
            mae_val.extend([val for _ in range(val_freq)])

        line_mae.set_data(range(len(mae)), mae)
        line_mae_val.set_data(range(len(mae_val)), mae_val)
        axs.set_xlim(0, len(mae))
        center_num = epoch_chunk_size
        tmp_min = min([min(mae[-center_num:]), min(mae_val[-center_num:])])
        tmp_max = max([max(mae[-center_num:]), max(mae_val[-center_num:])])
        axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

        plt.pause(1e-3)
        if len(hist.history['loss']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(mae), min(mae_val)])
    tmp_max = max([max(mae), max(mae_val)])
    axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

    plt.pause(1e-3)
    plt.savefig(f'{name}_loss_stage_2.png')

    opt = config.get('optimizer_2', optimizers.Adam(lr=1e-4))
    model.compile(
        optimizer=opt,
        loss="mae",  # mean_relative_mae,
        metrics=['mean_absolute_error', max_relative_mae]
        # loss='mean_absolute_error',
        # metrics=['mean_absolute_error']
    )

    for ini_epoch, epoch in zip(chunks[:-1], chunks[1:]):
        hist = model.fit(
            train_f,
            train_t,
            initial_epoch=ini_epoch,
            epochs=epoch,
            batch_size=len(train_t),
            validation_data=(test_f, test_t),
            validation_freq=val_freq,
            # callbacks=[early_stop],
            verbose=False,
        )

        mae.extend(hist.history['max_relative_mae'])

        for val in hist.history['val_max_relative_mae']:
            mae_val.extend([val for _ in range(val_freq)])

        line_mae.set_data(range(len(mae)), mae)
        line_mae_val.set_data(range(len(mae_val)), mae_val)
        axs.set_xlim(0, len(mae))
        center_num = epoch_chunk_size
        tmp_min = min([min(mae[-center_num:]), min(mae_val[-center_num:])])
        tmp_max = max([max(mae[-center_num:]), max(mae_val[-center_num:])])
        axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

        plt.pause(1e-3)
        if len(hist.history['loss']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(mae), min(mae_val)])
    tmp_max = max([max(mae), max(mae_val)])
    axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

    plt.pause(1e-3)
    plt.savefig(f'{name}_loss_stage_3.png')
    fig = plt.figure(2, figsize=(16, 10))
    fig.clf()
    fig, axss = plt.subplots(2, len(t.columns), figsize=(16, 10), num=2)
    test_pred = model.predict(test_f)
    train_pred = model.predict(train_f)

    for true, pred, ax, target in zip(test_t.T, test_pred.T, axss[1, :], t.columns):
        evaluate(true, pred, 'test', target, ax=ax)
        ret_dict[f'test {target} mae'] = skm.mean_absolute_error(true, pred)
        ret_dict[f'test {target} R2'] = skm.r2_score(true, pred)
        ret_dict[f'test {target} mean relative mae'] = np_mean_relative_mae(true, pred)
        ret_dict[f'test {target} max relative mae'] = np_max_relative_mae(true, pred)

    for true, pred, ax, target in zip(train_t.T, train_pred.T, axss[0, :], t.columns):
        evaluate(true, pred, 'train', target, ax=ax)
        ret_dict[f'train {target} mae'] = skm.mean_absolute_error(true, pred)
        ret_dict[f'train {target} R2'] = skm.r2_score(true, pred)
        ret_dict[f'train {target} mean relative mae'] = np_mean_relative_mae(true, pred)
        ret_dict[f'train {target} max relative mae'] = np_max_relative_mae(true, pred)

    model.save(f'{name}.h5')
    plt.savefig(f'{name}.png')

    return ret_dict


if __name__ == '__main__':
    all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    drop_target = {'NMHC(GT)'}

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target)
    )

    train_f_df, test_f_df, train_t_df, test_t_df = train_test_split(f, t, test_size=0.3, random_state=1203)
    train_f, test_f, train_t, test_t = train_f_df.values, test_f_df.values, train_t_df.values, test_t_df.values
    # model_name = 'results/256_3_relu'
    train_t_df.to_csv(f'train_target.csv', index=False)
    train_f_df.to_csv(f'train_features.csv', index=False)
    test_t_df.to_csv(f'test_target.csv', index=False)
    test_f_df.to_csv(f'test_features.csv', index=False)

    input_layer = layers.Input(shape=(f.shape[1],), dtype='float', name='Sensor_data')
    first_layer = layers.BatchNormalization(name='Preproprocessing')
    last_layer = layers.Dense(t.shape[1], 'linear', name='Output_layer')
    base_models = [
        {
            'name': os.path.join('results', 'relu_32_kreg'),
            'model': [
                first_layer,
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ],
        },
        {
            'name': os.path.join('results', 'relu_64_kreg'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_128_kreg'),
            'model': [
                first_layer,
                layers.Dense(
                    128,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_256_kreg'),
            'model': [
                first_layer,
                layers.Dense(
                    256,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    128,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_32_areg'),
            'model': [
                first_layer,
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ],
        },
        {
            'name': os.path.join('results', 'relu_64_areg'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_128_areg'),
            'model': [
                first_layer,
                layers.Dense(
                    128,
                    activation='relu',
                    name='Layer_1',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_2',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_256_areg'),
            'model': [
                first_layer,
                layers.Dense(
                    256,
                    activation='relu',
                    name='Layer_1',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    128,
                    activation='relu',
                    name='Layer_2',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_3',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        }
    ]

    try:
        results = pd.read_csv('results_prev.csv')
    except:
        results = pd.DataFrame()

    configs_noise = list()
    noises = [0.05, 0.5]
    lrs = [1e-3, 5e-4, 1e-4]
    tot_num = len(noises) * len(lrs) * len(base_models)
    curr = 0
    for noise in noises:
        for lr in lrs:
            for model in base_models:
                tmp_config = copy.deepcopy(model)
                tmp_config['name'] = f"{tmp_config['name']}_GN{noise}"
                tmp_config['name'] = f"{tmp_config['name']}_lr{lr}"
                tmp_config['model'].insert(1, layers.GaussianNoise(noise, name='Noise'))
                tmp_config['model'].insert(0, input_layer)
                tmp_config['optimizer'] = optimizers.Adam(lr=lr)
                tmp_config['optimizer_2'] = optimizers.Adam(lr=lr / 10)
                print(f'{curr}/{tot_num}')
                results = results.append(train_model(tmp_config), ignore_index=True)
                print(results)
                results.to_csv('temp_results.csv', index=False)
                curr += 1
    results.to_csv('results.csv', index=False)
