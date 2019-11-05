import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

header = [
  "eventNumber", "weight",
  "ljet1_pt", "ljet1_eta", "ljet1_phi", "ljet1_E", "ljet1_M",
  "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_E", "ljet2_M",
  "jj_pt",    "jj_eta",    "jj_phi",    "jj_E",    "jj_M",
  "jj_dPhi",  "jj_dEta",  "jj_dR",
]
features = [
    "ljet1_pt", "ljet1_eta", "ljet1_M",
    "ljet2_pt", "ljet2_eta", "ljet2_phi", "ljet2_M"
]

infilename = "csv/mg5_dijet_ht500.ptcl.pt250.nominal.csv"
level = infilename.split("/")[-1].split('.')[1]
data = pd.read_csv(infilename, delimiter=',', names=header)[features]
data.dropna(inplace=True)
X_train = data.values

scaler = MinMaxScaler((-1, 1))
scaler.fit(X_train)

scaler_path = Path('GAN') / f'scaler.{level}.pkl'
with scaler_path.open("wb") as file_scaler:
    pickle.dump(scaler, file_scaler)
print (f'scaler saved to {scaler_path}')
