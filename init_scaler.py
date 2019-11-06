import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data import PTCL_HEADER, PTCL_FEATURES


def main():
    infilename = "csv/mg5_dijet_ht500.ptcl.pt250.nominal.csv"
    level = infilename.split("/")[-1].split('.')[1]
    data = pd.read_csv(infilename, delimiter=',', names=PTCL_HEADER)[PTCL_FEATURES]
    data.dropna(inplace=True)
    X_train = data.values

    scaler = MinMaxScaler((-1, 1))
    scaler.fit(X_train)

    scaler_path = Path(f'scaler.{level}.pkl')
    with scaler_path.open("wb") as file_scaler:
        pickle.dump(scaler, file_scaler)
    print(f'scaler saved to {scaler_path}')


if __name__ == '__main__':
    main()