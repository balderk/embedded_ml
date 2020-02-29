from data.get_data import get_feature_targets
from evaluation.simple_evaluation import evaluate

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    drop_target = {'NMHC(GT)'}

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target)
    )

    train_f_df, test_f_df, train_t_df, test_t_df = train_test_split(f, t, test_size=0.3, random_state=1203)
    train_f, test_f, train_t, test_t = train_f_df.values, test_f_df.values, train_t_df.values, test_t_df.values
    model_name = 'results/256_3_relu'
    train_t_df.to_csv(f'{model_name}_train_target.csv', index=False)
    train_f_df.to_csv(f'{model_name}_train_features.csv', index=False)
    test_t_df.to_csv(f'{model_name}_test_target.csv', index=False)
    test_f_df.to_csv(f'{model_name}_test_features.csv', index=False)

    print(test_f_df.shape)
    print(test_t_df.shape)
    model = Sequential([
        layers.BatchNormalization(input_shape=(f.shape[1],)),
        layers.GaussianNoise(2),
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1)
        ),
        layers.Dense(256, activation='relu'),
        layers.GaussianNoise(.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(t.shape[1], 'linear')
    ])

    opt = optimizers.Adam(lr=1e-3)
    model.compile(
        optimizer=opt,
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']  # ['mae', 'mse']
    )

    fig, axs = plt.subplots(1, 1)

    mae = []
    mae_val = []

    line_mae = axs.plot(mae, range(len(mae)), 'b-', label='mae')[0]

    line_mae_val = axs.plot(mae_val, range(len(mae_val)), 'r-', label='mae_val')[0]

    axs.grid()
    axs.set_title('mean_absolute_error')
    axs.legend()
    plt.pause(1e-3)
    total_epoch = 100000
    val_freq = 10
    epoch_chunk_size = 1000
    chunks = np.arange(0, total_epoch, epoch_chunk_size, dtype='int')
    early_stop = EarlyStopping(monitor='val_mean_absolute_error', patience=50)
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
        )

        mae.extend(hist.history['mean_absolute_error'])

        for val in hist.history['val_mean_absolute_error']:
            mae_val.extend([val for _ in range(val_freq)])

        line_mae.set_data(range(len(mae)), mae)
        line_mae_val.set_data(range(len(mae_val)), mae_val)
        axs.set_xlim(0, len(mae))
        center_num = 5000
        tmp_min = min([min(mae[-center_num:]), min(mae_val[-center_num:])])
        tmp_max = max([max(mae[-center_num:]), max(mae_val[-center_num:])])
        axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

        plt.pause(1e-3)
        if len(hist.history['mean_absolute_error']) != epoch_chunk_size:
            # early stopping
            break

    tmp_min = min([min(mae), min(mae_val)])
    tmp_max = max([max(mae), max(mae_val)])
    axs.set_ylim(min([tmp_min * 0.9, tmp_min - 1]), max([tmp_max * 1.1, tmp_max + 1]))

    plt.pause(1e-3)
    plt.savefig(model_name+'loss.png')
    fig, axs = plt.subplots(2, len(t.columns), figsize=(16, 10))
    test_pred = model.predict(test_f)
    train_pred = model.predict(train_f)

    for true, pred, ax, target in zip(test_t.T, test_pred.T, axs[1, :], t.columns):
        evaluate(true, pred, 'test', target, ax=ax)

    for true, pred, ax, target in zip(train_t.T, train_pred.T, axs[0, :], t.columns):
        evaluate(true, pred, 'train', target, ax=ax)


    model.save(f'{model_name}.h5')
    plt.savefig(f'{model_name}.png')

