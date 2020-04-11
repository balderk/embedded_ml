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
from tensorflow.keras.losses import mean_absolute_percentage_error

import tensorflow.keras.backend as kb

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.utils import model_to_dot, plot_model


def mean_absolute_relative_error(y_actual, y_pred) -> tf.Tensor:
    return kb.mean(kb.abs(y_actual - y_pred) / y_actual)


def max_absolute_relative_error(y_actual, y_pred) -> tf.Tensor:
    return kb.max(kb.abs(y_actual - y_pred) / y_actual)


def np_mean_absolute_relative_error(y_actual, y_pred):
    return np.mean(np.abs(y_actual - y_pred) / y_actual)


def np_max_absolute_relative_error(y_actual, y_pred):
    return np.max(np.abs(y_actual - y_pred) / y_actual)


def train_model(config: dict) -> dict:
    # getting the model definitions
    name = config.get('name', None)
    model_definition = config.get('model', None)
    ret_dict = {'name': name}
    opt = config.get('optimizer', optimizers.Adam(lr=1e-3))

    model = Sequential(model_definition)  # creating the Keras model object
    model.compile(
        optimizer=opt,
        loss=mean_absolute_percentage_error,
        metrics=['mean_absolute_error', mean_absolute_relative_error]
    )

    plot_model(model, to_file=f'{name}_model.png', show_shapes=True, expand_nested=True)
    thing = model_to_dot(model, show_shapes=True, expand_nested=True).create(prog='dot',
                                                                             format='svg')  # .create_svg(f'{name}.svg')
    with open(f'{name}.svg', 'wb') as fil:
        fil.write(thing)

    # Creating the plots
    fig = plt.figure(1)
    fig.clf()
    fig, axs = plt.subplots(1, 1, num=1)
    loss = []
    loss_val = []
    line_loss = axs.plot(loss, range(len(loss)), 'b-', label='loss')[0]
    line_loss_val = axs.plot(loss_val, range(len(loss_val)), 'r-', label='loss_val')[0]
    axs.grid()
    axs.set_title(f'{name} loss')
    axs.legend()
    plt.pause(1e-3)

    # Configuring the batches
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

        loss.extend(hist.history['loss'])

        for val in hist.history['val_loss']:
            loss_val.extend([val for _ in range(val_freq)])

        line_loss.set_data(range(len(loss)), loss)
        line_loss_val.set_data(range(len(loss_val)), loss_val)
        axs.set_xlim(0, len(loss))
        center_num = epoch_chunk_size
        tmp_min = min([min(loss[-center_num:]), min(loss_val[-center_num:])])
        tmp_max = max([max(loss[-center_num:]), max(loss_val[-center_num:])])
        axs.set_ylim(max(min([tmp_min * 0.9, tmp_min - 1]), 0), max([tmp_max * 1.1, tmp_max + 1]))
        axs.set_ylim(max(min([tmp_min * 0.9, tmp_min - 1]), 0), min(max([tmp_max * 1.1, tmp_max + 1]), max(loss_val)))

        plt.pause(1e-3)
        if len(hist.history['loss']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(loss), min(loss_val)])
    tmp_max = max([max(loss), max(loss_val)])
    axs.set_ylim(max(min([tmp_min * 0.9, tmp_min - 1]), 0), min(max([tmp_max * 1.1, tmp_max + 1]), max(loss_val)))

    plt.pause(1e-3)
    plt.savefig(f'{name}_loss_stage_1.png')

    opt = config.get('optimizer_2', optimizers.Adam(lr=1e-4))
    model.compile(
        optimizer=opt,
        loss=mean_absolute_percentage_error,
        metrics=['mean_absolute_error', mean_absolute_relative_error]
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

        loss.extend(hist.history['loss'])

        for val in hist.history['val_loss']:
            loss_val.extend([val for _ in range(val_freq)])

        line_loss.set_data(range(len(loss)), loss)
        line_loss_val.set_data(range(len(loss)), loss_val)
        axs.set_xlim(0, len(loss))
        center_num = epoch_chunk_size
        tmp_min = min([min(loss[-center_num:]), min(loss_val[-center_num:])])
        tmp_max = max([max(loss[-center_num:]), max(loss_val[-center_num:])])
        axs.set_ylim(max(min([tmp_min * 0.9, tmp_min - 1]), 0), min(max([tmp_max * 1.1, tmp_max + 1]), max(loss_val)))

        plt.pause(1e-3)
        if len(hist.history['loss']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(loss), min(loss_val)])
    tmp_max = max([max(loss), max(loss_val)])
    axs.set_ylim(max(min([tmp_min * 0.9, tmp_min - 1]), 0), min(max([tmp_max * 1.1, tmp_max + 1]), max(loss_val)))

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
        ret_dict[f'test {target} mean relative mae'] = np_mean_absolute_relative_error(true, pred)
        ret_dict[f'test {target} max relative mae'] = np_max_absolute_relative_error(true, pred)

    for true, pred, ax, target in zip(train_t.T, train_pred.T, axss[0, :], t.columns):
        evaluate(true, pred, 'train', target, ax=ax)
        ret_dict[f'train {target} mae'] = skm.mean_absolute_error(true, pred)
        ret_dict[f'train {target} R2'] = skm.r2_score(true, pred)
        ret_dict[f'train {target} mean relative mae'] = np_mean_absolute_relative_error(true, pred)
        ret_dict[f'train {target} max relative mae'] = np_max_absolute_relative_error(true, pred)

    model.save(f'{name}.h5')
    plt.savefig(f'{name}.png')

    return ret_dict


if __name__ == '__main__':
    all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    drop_target = {'NMHC(GT)'}

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target),
        drop_outliers=True
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
            'name': os.path.join('results', 'relu_64_kareg_mare'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01)
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_64_mare'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_64_kreg_mare'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                    kernel_regularizer=regularizers.l2(0.01),
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_2',
                    kernel_regularizer=regularizers.l2(0.01),
                ),
                layers.Dense(
                    32,
                    activation='relu',
                    name='Layer_3',
                    kernel_regularizer=regularizers.l2(0.01),
                ),
                last_layer,
            ]
        },
        {
            'name': os.path.join('results', 'relu_64_areg_mare'),
            'model': [
                first_layer,
                layers.Dense(
                    64,
                    activation='relu',
                    name='Layer_1',
                    activity_regularizer=regularizers.l2(0.01)
                ),
                layers.Dense(
                    32,
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
        }
    ]

    try:
        results = pd.read_csv('temp_results.csv')
    except:
        results = pd.DataFrame()

    configs_noise = list()
    noises = [0.005, 0.01, 0.05, 0.1]
    lrs = [1e-3, 1e-4]
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

                if tmp_config['name'] not in results['name']:
                    results = results.append(train_model(tmp_config), ignore_index=True)
                print(results)
                results.to_csv('temp_results.csv', index=False)
                curr += 1
    results.to_csv('results.csv', index=False)
