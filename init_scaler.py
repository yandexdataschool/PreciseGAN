import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data import PTCL_HEADER, PTCL_FEATURES


def main(args):
    level = args.csv_path.split("/")[-1].split('.')[1]
    data = pd.read_csv(args.csv_path, delimiter=',', names=PTCL_HEADER, usecols=PTCL_FEATURES)
    X_train = data.values

    scaler = MinMaxScaler((-1, 1))
    scaler.fit(X_train)

    scaler_path = Path(f'scaler.{level}.pkl')
    with scaler_path.open("wb") as file_scaler:
        pickle.dump(scaler, file_scaler)
    print(f'scaler saved to {scaler_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
