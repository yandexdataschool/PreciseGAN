import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import split_data
from util import fix_seed


PTCL_HEADER = [
  "eventNumber", "weight",
  "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
  "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
  "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
  "jj_dPhi",  "jj_dEta",  "jj_dR",
]

PTCL_FEATURES = [
    "ljet1_pt", "ljet1_eta", "ljet1_M",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M"
]


def main(args):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    fix_seed(args.seed)
    if args.csv_path is None:
        args.csv_path = "csv/%s.%s.%s.%s.csv" % (args.dsid, args.level, args.preselection, args.systematic)
    logging.info(f'training file: {args.csv_path}')

    data = pd.read_csv(args.csv_path, delimiter=',', names=PTCL_HEADER)
    data = data[PTCL_FEATURES]
    items = data.values

    logging.info(f'input features: {list(data.columns)}')
    logging.info(f'total number of input features: {len(data.columns)}')

    train_indices, test_indices = split_data(items, args.train_split, args.random_split)

    source_path = Path(args.csv_path)
    parent_path = source_path.parent
    file_name = source_path.stem

    train_file = parent_path / f'train_{file_name}'
    test_file = parent_path / f'test_{file_name}'

    logging.info(f'saving train/test files to {str(train_file)} and {str(test_file)}...')

    np.save(train_file, items[train_indices])
    np.save(test_file, items[test_indices])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--train_split', type=float, default=0.15)
    parser.add_argument('--random_split', type=bool, default=False)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('-l', '--level', default="ptcl")
    parser.add_argument('-p', '--preselection', default="pt250")
    parser.add_argument('-s', '--systematic', default="nominal")
    parser.add_argument('-d', '--dsid', default="mg5_dijet_ht500")

    args = parser.parse_args()

    main(args)
