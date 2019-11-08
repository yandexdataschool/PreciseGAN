import logging
import pickle

import numpy as np
from torch.utils.data.dataset import Dataset


class DiJetDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)

    @classmethod
    def from_path(cls, path, scaler=None):
        items = np.load(path)

        if scaler is not None:
            items = scaler.transform(items)

        return cls(items)


def get_data(args):
    scaler_filename = "scaler.%s.pkl" % args.level if args.scaler_dump is None else args.scaler_dump
    logging.info(f'loading scaler from {scaler_filename}')
    with open(scaler_filename, "rb") as file_scaler:
        scaler = pickle.load(file_scaler)

    dataset_train = DiJetDataset.from_path(args.train_data, scaler)
    dataset_test = DiJetDataset.from_path(args.test_data)

    return dataset_train, dataset_test, scaler


PTCL_HEADER = [
  "eventNumber", "weight",
  "Leading large-R jet p_t", "Leading large-R jet η", "ljet1_phi", "ljet1_E", "Leading large-R jet m",
  "2nd leading large-R jet p_t", "2nd leading large-R jet η", "ljet2_phi", "ljet2_E", "2nd leading large-R jet m",
  "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
  "jj_dPhi",  "jj_dEta",  "jj_dR",
]
PTCL_FEATURES = [
    "Leading large-R jet p_t", "Leading large-R jet η", "Leading large-R jet m",
    "2nd leading large-R jet p_t", "2nd leading large-R jet η", "ljet2_phi", "2nd leading large-R jet m"
]
DIJET_SYSTEM_FEATURES = [
    "Dijet system pt", "Dijet system η", "Dijet system m"
]