import os
import tensorflow.keras as keras
from data.get_data import get_feature_targets
from evaluation.simple_evaluation import evaluate
import tensorflow.keras.optimizers as optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
drop_target = {'NMHC(GT)'}

model_name = os.path.join('results', '256_relu.h5')

model = keras.models.load_model(model_name)
f, t = get_feature_targets(
    dropna=True,
    target_keys=list(set(all_target_keys) - drop_target)
)

train_f, test_f, train_t, test_t = train_test_split(f.values, t.values, test_size=0.3, random_state=1203)
opt = optimizers.Adam(lr=1e-5)
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
val_freq = 100
chunks = np.arange(0, total_epoch, 1000, dtype='int')
for ini_epoch, epoch in zip(chunks[:-1], chunks[1:]):
    hist = model.fit(
        train_f,
        train_t,
        initial_epoch=ini_epoch,
        epochs=epoch,
        batch_size=len(train_t),
        validation_data=(test_f, test_t),
        validation_freq=val_freq,
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

fig, axs = plt.subplots(2, len(t.columns), figsize=(16, 10))
test_pred = model.predict(test_f)
train_pred = model.predict(train_f)

for true, pred, ax, target in zip(test_t.T, test_pred.T, axs[1, :], t.columns):
    evaluate(true, pred, 'test', target, ax=ax)

for true, pred, ax, target in zip(train_t.T, train_pred.T, axs[0, :], t.columns):
    evaluate(true, pred, 'train', target, ax=ax)

model.save(f'{model_name}')
plt.savefig(f'{model_name}.png')
