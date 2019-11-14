import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from data import PTCL_HEADER, PTCL_FEATURES


def main(args):
    level = args.csv_path.split("/")[-1].split('.')[1]
    data = pd.read_csv(args.csv_path, delimiter=',', names=PTCL_HEADER)

    if (args.task == 'tail'):
        data = data[data['jj_M'] > 1500]

    data = data[PTCL_FEATURES]
    X_train = data.values

    if args.scaler == 'minmax':
        scaler = MinMaxScaler((-1, 1))
    elif args.scaler == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        raise ValueError(f'Unknown scaler type: {args.scaler}')

    scaler.fit(X_train)

    scaler_path = Path(f'scaler.{level}.pkl')
    with scaler_path.open("wb") as file_scaler:
        pickle.dump(scaler, file_scaler)
    print(f'scaler saved to {scaler_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('-t', '--task', default='integral', choices={'integral', 'tail'})
    parser.add_argument('-s', '--scaler', default='minmax', choices={'minmax', 'quantile'})
    args = parser.parse_args()
    main(args)
