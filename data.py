import logging
import math
import pickle
import random
from pathlib import Path

import pandas as pd
from torch.utils.data.dataset import Dataset

header = [
  "eventNumber", "weight",
  "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
  "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
  "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
  "jj_dPhi",  "jj_dEta",  "jj_dR",
]


class DiJetDataset(Dataset):
    features = [
        "ljet1_pt", "ljet1_eta", "ljet1_M",
        "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M"
    ]

    def __init__(self, items):
        self.items = items

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)

    @classmethod
    def from_path(cls, path, scaler=None):
        data = pd.read_csv(path, delimiter=',', names=header)
        data = data[cls.features]
        items = data.values

        if scaler is not None:
            items = scaler.transform(items)

        logging.info(f'input features: {list(data.columns)}')
        logging.info(f'total number of input features: {len(data.columns)}')

        return cls(items)

    @staticmethod
    def get_cached(path, scaler=None):
        cached_object = str(path.name) + '_cached'
        if Path(cached_object).exists():
            logging.info(f'loading cached object from {cached_object}')
            with path.open('rb') as f:
                return pickle.load(f)
        path.parent.mkdir(exist_ok=True)
        dataset = DiJetDataset.from_path(str(path), scaler)

        logging.info(f'saved cached object to {cached_object}')
        with (Path(cached_object)).open(mode='wb') as f:
            pickle.dump(dataset, f)

        return dataset


def split_data(dataset, train_split, shuffle=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(math.floor(train_split * dataset_size))
    if shuffle:
        random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    return train_indices, val_indices
