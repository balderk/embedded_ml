from data.get_data import get_data
import pandas as pd
pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
import missingno
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = get_data()

    print(data.describe())
    missingno.matrix(data, fontsize=12)
    plt.savefig('data_information.svg')
    plt.savefig('data_information.png')