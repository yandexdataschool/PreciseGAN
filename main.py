import argparse
import logging
import pickle

import torch
from comet_ml import Experiment
from torch.utils.data.dataset import Subset

from data import DiJetDataset, split_data
from eval import evaluate_model
from model import GeneratorCNN, DiscriminatorCNN
from optim import setup_optimizer
from train import train
from util import fix_seed


def main_train(args):
    fix_seed(args.seed)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    if args.training_filename is None:
        args.training_filename = "csv/%s.%s.%s.%s.csv" % (args.dsid, args.level, args.preselection, args.systematic)
        logging.info(f'training file: {args.training_filename}')
    else:
        args.systematic = args.training_filename.split("/")[-1].split('.')[-2]

    logging.info(f'training level: {args.level}')
    logging.info(f'training systematic: {args.systematic}')

    scaler_filename = "scaler.%s.pkl" % args.level
    logging.info(f'loading scaler from {scaler_filename}')
    with open(scaler_filename, "rb") as file_scaler:
        scaler = pickle.load(file_scaler)

    dataset = DiJetDataset.from_path(args.training_filename, scaler)

    n_features = len(DiJetDataset.features)

    generator = GeneratorCNN(args.gan_noise_size, n_features).to(device)
    discriminator = DiscriminatorCNN(n_features).to(device)

    optimizer_d = setup_optimizer(discriminator, args.learning_rate, weight_decay=0)
    optimizer_g = setup_optimizer(generator, args.learning_rate, weight_decay=0)

    experiment = Experiment('gflIAsawYkIJvtkFb55lOwno7', project_name="sirius-gan-tails", workspace="v3rganz")
    experiment.log_parameters(vars(args))

    train_indices, val_indices = split_data(dataset, 0.15, True)

    train(generator, discriminator, args, Subset(dataset, train_indices), optimizer_g, optimizer_d, scaler=scaler,
          test_dataset=dataset.items[val_indices][:len(val_indices) // 10],
          ecaluate_every=args.log_every, experiment=experiment, device=device)

    n_events = len(dataset)
    steps = n_events // 512

    evaluate_model(generator, experiment, dataset.items[val_indices], 512, steps, args.gan_noise_size, device, scaler)
    experiment.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ttbar diffxs sqrt(s) = 13 TeV classifier training')
    parser.add_argument('-i', '--training_filename')
    parser.add_argument('-l', '--level', default="reco")
    parser.add_argument('-p', '--preselection', default="pt250")
    parser.add_argument('-s', '--systematic', default="nominal")
    parser.add_argument('-d', '--dsid', default="mg5_dijet_ht500")
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-tr', '--training_ratio', type=int, default=1)
    parser.add_argument('-le', '--log_every', type=int, default=500)
    parser.add_argument('-n', '--gan_noise_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=48)
    args = parser.parse_args()

    main_train(args)
