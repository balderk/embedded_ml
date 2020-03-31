import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    df = pd.read_csv('altered_results.csv')

    name_col = ['name']
    test_mare_cols = [v for v in df.columns if 'test' in v and 'relative' in v and 'mean' in v]
    test_rae_cols = [v for v in df.columns if 'test' in v and 'rae' in v and '90' in v]
    test_r2_cols = [v for v in df.columns if 'test' in v and 'R2' in v]
    cols_to_use = name_col + test_mare_cols + test_rae_cols + test_r2_cols
    print(df.describe())
    print(cols_to_use)
    print(df[cols_to_use])
    print(df[cols_to_use].describe())

    new_df = df[cols_to_use]
    new_df.loc[:, 'mare sum'] = new_df.loc[:, test_mare_cols].sum(axis=1)
    new_df.loc[:, 'R2 sum'] = new_df.loc[:, test_r2_cols].sum(axis=1)
    print(new_df.sort_values('mare sum', ascending=True).head(20))
    print(new_df.sort_values('R2 sum', ascending=False).head(20))
