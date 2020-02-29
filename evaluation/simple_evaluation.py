from data.get_data import get_feature_targets

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from sklearn.tree import ExtraTreeRegressor

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.svm import SVR


def evaluate(true, pred, name='', target='', ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    print(f'{target:10}|{name:10}| R2: {r2:>6.2f}, mae: {mae:>6.2f}  ')
    ax.set_title(f'{target} {name} R2: {r2:.2f}')
    ax.scatter(true, pred)
    ax.set_xlabel('true')
    ax.set_ylabel('pred')
    ax.grid()


if __name__ == '__main__':
    all_target_keys = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    drop_target = {'NMHC(GT)'}

    f, t = get_feature_targets(
        dropna=True,
        target_keys=list(set(all_target_keys) - drop_target)
    )

    train_f, test_f, train_t_all, test_t_all = train_test_split(f, t, test_size=0.3, random_state=123)
    pca = PCA()

    pca.fit(train_f)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    for x1, x2, name in zip(pc1, pc2, [n for n in test_f]):
        plt.plot(x1, x2, '.', label=name)
    plt.legend()
    plt.grid()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.grid()
    plt.show()

    pca = PCA(n_components=4)
    train_f_red = pca.fit_transform(train_f)
    test_f_red = pca.transform(test_f)
    for model in [LinearRegression(), SVR(gamma='scale'), GradientBoostingRegressor(), ExtraTreeRegressor()]:
        model_name = model.__repr__().split('(')[0]
        fig, axs = plt.subplots(2, len(train_t_all.columns), figsize=(16, 10))

        fig.suptitle(model_name)
        for target, (ax1, ax2) in zip(train_t_all, axs.T):
            train_t = train_t_all[target]
            test_t = test_t_all[target]
            lr = model #LinearRegression()
            lr.fit(train_f_red, train_t)
            train_pred = lr.predict(train_f_red)
            test_pred = lr.predict(test_f_red)

            evaluate(train_t, train_pred, 'train', target, ax=ax1)
            # plt.show()
            evaluate(test_t, test_pred, 'test', target, ax=ax2)

        # plt.tight_layout()
        plt.savefig(f'{model_name}.png')
        plt.show()

        #plt.plot(range(len(train_t)),
        #         train_t - train_pred,
        #         '.', label='train_residual')
        #plt.plot(range(len(train_t), len(train_t) + len(test_t)),
        #         test_t - test_pred,
        #         '.', label='test_residual')
        #plt.title('residuals')
        #plt.legend()
        #plt.grid()
        #plt.show()
