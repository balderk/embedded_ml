
from data.get_data import get_feature_targets

import hoggorm as ho
import hoggormplot as hopl

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    drop_target = {'NMHC(GT)'}

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target),
        drop_outliers=True
    )

    values = pd.concat([f, t], axis=1)

    model = ho.nipalsPCA(arrX=values.values)
    y = np.linalg.norm(model.X_scores(), axis=1, ord=1)
    x = np.arange(len(y))
    df = pd.DataFrame({'idx':x.T, 'val':y.T})
    plt.plot(sorted(y), '.')
    plt.grid()
    print(', '.join([f'{v}' for v in df.loc[df['val'] > 3100]['idx'].values]))
    #hopl.scores(model, objNames=['']*f.shape[0])

    np.linalg.norm