# PreciseGAN
A research repo for studying different techniques towards making more precise GANs

## Experiment reproduction

To reproduce experiment first you need data, that can be downloaded with `get_csv.sh` script.
Then you should learn and dump `MinMaxScaler` with script `init_scaler.py`, and make train/test data split
with `init_data.py`

Then run `main.py` with preferred parameters

Notebook usage example: `notebook_usage.ipynb`

### Original paper reproduction

To reproduce results of original paper __"DijetGAN: a Generative-Adversarial Network approach for the simulation of QCD
dijet events at the LHC"__ you should initialize data with `--train_split=0.8` for integral training data and 
`--train_split=0.15 --task=tail` for tail training data

then run `main.py` with CLI parameters `--architecture=cnn --iterations=500000 --optim=adam -lr 1e-5 --adam_beta_1=0.5 
--adam_beta_2=0.9 --batch_size=128` and `--level=ptcl` for particle level or `--level=reco` for reco level, also specify
`--task=integral` for integral training or `--task=tail` for tail training.
