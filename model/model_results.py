import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    df = pd.read_csv('altered_results.csv')

    look_for = '64'

    name_col = ['name']
    test_mape_cols = [v for v in df.columns if 'test' in v and 'percentage' in v and 'mean' in v]
    test_ape_cols = [v for v in df.columns if 'test' in v and 'ape' in v and 'mean' in v]
    test_mae_cols = [v for v in df.columns if 'test' in v and 'mae' in v]
    test_r2_cols = [v for v in df.columns if 'test' in v and 'R2' in v]
    cols_to_use = name_col + test_mape_cols + test_ape_cols + test_r2_cols + test_mae_cols
    print(df.describe())
    print(cols_to_use)
    print(df[cols_to_use])
    print(df[cols_to_use].describe())

    rows_to_use = [i for i, v in enumerate(df['name']) if look_for in v]

    new_df = df.loc[rows_to_use, cols_to_use]
    new_df['name'] = [v.replace('results/', '') for v in new_df['name']]
    new_df.loc[:, 'mape mean'] = new_df.loc[:, test_mape_cols].mean(axis=1)
    new_df.loc[:, 'mae mean'] = new_df.loc[:, test_mae_cols].mean(axis=1)
    new_df.loc[:, 'R2 mean'] = new_df.loc[:, test_r2_cols].mean(axis=1)
    print()

    print('mape table')
    disp_df = new_df.loc[:, ['name', 'mape mean', 'mae mean', 'R2 mean']]
    disp_df.sort_values('mape mean', ascending=True).head(20).to_csv('summary/mape_result.csv')
    print(disp_df.sort_values('mape mean', ascending=True).head(20))
    print()

    print('R2 table')
    disp_df.sort_values('R2 mean', ascending=False).head(20).to_csv('summary/R2_result.csv')
    print(disp_df.sort_values('R2 mean', ascending=False).head(20))
    print()

    print('mae table')
    disp_df.sort_values('mae mean', ascending=True).head(20).to_csv('summary/mae_result.csv')
    print(disp_df.sort_values('mae mean', ascending=True).head(20))
    print()
