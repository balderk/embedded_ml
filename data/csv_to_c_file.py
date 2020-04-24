import pandas as pd
import numpy
import os

_H_FILE_HEADER = """
#pragma clang diagnostic push
#pragma ide diagnostic ignored 
/**
This file is generated by "csv_to_c_file.py", written by Balder G. Klanderud
*/
#ifndef __{name}
#define __{name}

#define {NAME}_NUM_ROWS\t {rows}
#define {NAME}_NUM_COLS\t {cols}

const char * {name}_column_description[{NAME}_NUM_COLS] = {start_bracket} {col_description} {stop_bracket};

const {dtype} {name} [{NAME}_NUM_ROWS][{NAME}_NUM_COLS] = {start_bracket}
"""

_H_FILE_BOTTOM = """{stop_bracket};
#endif
#pragma clang diagnostic pop
"""

_numpy_to_ctype = {
    'float64': 'double',
    'float32': 'float',
}


def df_to_c_file(df: pd.DataFrame, name: str, path='', convert_to_float=True):
    h_filename = f'{name}.h'
    h_filepath = os.path.join(path, h_filename)
    np_arr: numpy.array = df.values
    print(h_filepath)
    if convert_to_float:
        np_arr = np_arr.astype('float32')

    rows, cols = np_arr.shape
    dtype = _numpy_to_ctype[np_arr.dtype.name]
    stuff_dict = {
        'name': name,
        'NAME': name.upper(),
        'h_filename': h_filename,
        'h_filepath': h_filepath,
        'rows': rows,
        'cols': cols,
        'dtype': dtype,
        'stop_bracket': '}',
        'start_bracket': '{',
        'col_description': ', '.join([f'"{val}"' for val in df.columns])
    }
    with open(h_filepath, 'w+') as h_file:
        h_file.write(_H_FILE_HEADER.format(**stuff_dict))

        data = list()
        for row in np_arr:
            data.append(', '.join([f'{val}' for val in row]))

        data_str = '{' + '},\n{'.join(data) + '}\n'

        h_file.write(data_str)

        h_file.write(_H_FILE_BOTTOM.format(**stuff_dict))


if __name__ == '__main__':
    test_features_file = '/home/balder/PycharmProjects/embedded_ml/model/keras_tuner/keras_tuner_test_features.csv'
    test_target_file = '/home/balder/PycharmProjects/embedded_ml/model/keras_tuner/keras_tuner_test_target.csv'
    train_features_file = '/home/balder/PycharmProjects/embedded_ml/model/keras_tuner/keras_tuner_train_features.csv'
    train_target_file = '/home/balder/PycharmProjects/embedded_ml/model/keras_tuner/keras_tuner_train_target.csv'
    # test_features_file = '/home/balder/PycharmProjects/embedded_ml/model/results/512_relu_test_features.csv'
    # test_target_file = '/home/balder/PycharmProjects/embedded_ml/model/results/512_relu_test_target.csv'
    # train_features_file = '/home/balder/PycharmProjects/embedded_ml/model/results/512_relu_train_features.csv'
    # train_target_file = '/home/balder/PycharmProjects/embedded_ml/model/results/512_relu_train_target.csv'

    df_to_c_file(pd.read_csv(test_features_file), 'test_features', path='data_dir/')
    df_to_c_file(pd.read_csv(test_target_file), 'test_target', path='data_dir/')
    df_to_c_file(pd.read_csv(train_features_file), 'train_features', path='data_dir/')
    df_to_c_file(pd.read_csv(train_target_file), 'train_target', path='data_dir/')