import argparse
import logging
from datetime import datetime
from pathlib import Path

from comet_ml import Experiment
import torch

from data import get_data
from evaluation import evaluate_model
from hyperparam import RandInt, rand_search, RandChoice
from model import get_models
from optim import setup_optimizer
from train import train
from util import fix_seed, save_model, load_model


def main_random_search(args):
    args_generators = {
        'gan_noise_size': RandChoice([64, 128, 256]),
        'architecture': RandChoice(['cnn', 'fc']),
        'batch_size': RandInt(32, 512),
        'training_ratio': RandInt(1, 10)
    }

    rand_search(main_train, args, args_generators)


def main_eval(args):
    assert args.load_from is not None, '--load_from required in eval mode'

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    dataset_train, dataset_test, scaler = get_data(args)

    logging.info(f'evaluation mode. Level: {args.level}')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    n_features = dataset_train.items.shape[1]
    generator, discriminator = get_models(args, n_features, device)

    experiment = Experiment(args.comet_api_key, project_name=args.comet_project_name, workspace=args.comet_workspace)
    experiment.log_parameters(vars(args))

    load_model(Path(args.load_from), generator, discriminator, None, None, device)

    n_events = len(dataset_test)
    steps = n_events // args.eval_batch_size

    evaluate_model(generator, experiment, dataset_test, args.eval_batch_size, steps, args, device, scaler, 0)


def main_train(args):
    now = datetime.now()
    save_to = Path(args.save_to) if args.save_to is not None else Path().cwd()
    save_dir = save_to / f'{now:%Y%m%d-%H%M-%S}'
    fix_seed(args.seed)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train, dataset_test, scaler = get_data(args)

    logging.info(f'training level: {args.level}')

    n_features = dataset_train.items.shape[1]

    generator, discriminator = get_models(args, n_features, device)

    optimizer_d = setup_optimizer(discriminator, args.learning_rate, weight_decay=0, args=args)
    optimizer_g = setup_optimizer(generator, args.learning_rate, weight_decay=0, args=args)

    experiment = Experiment(args.comet_api_key, project_name=args.comet_project_name, workspace=args.comet_workspace)
    experiment.log_parameters(vars(args))
    iterations_total = train(generator, discriminator, args, dataset_train, optimizer_g, optimizer_d, scaler=scaler,
                           save_dir=save_dir, test_dataset=dataset_test.items[:len(dataset_test) // 10],
                           experiment=experiment, device=device)

    n_events = len(dataset_test)
    steps = n_events // args.eval_batch_size

    evaluate_model(generator, experiment, dataset_test, args.eval_batch_size, steps, args, device, scaler,
                   iterations_total)
    experiment.end()

    save_model(save_dir, generator, discriminator, optimizer_g, optimizer_d, iterations_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--scaler_dump')
    parser.add_argument('--save_to', type=str)
    parser.add_argument('--load_from', type=str)
    parser.add_argument('--mode', type=str, default='train', choices={'train', 'eval'})
    parser.add_argument('-a', '--architecture', default='cnn', choices={'cnn', 'fc'})
    parser.add_argument('-o', '--optim', default='sgd', choices={'sgd', 'adam'})
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.99)
    parser.add_argument('-l', '--level', default="ptcl")
    parser.add_argument('-e', '--iterations', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-tr', '--training_ratio', type=int, default=1)
    parser.add_argument('-le', '--log_every', type=int, default=500)
    parser.add_argument('-se', '--save_every', type=int, default=5000)
    parser.add_argument('-n', '--gan_noise_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--comet_api_key', type=str, required=True)
    parser.add_argument('--comet_project_name', type=str, required=True)
    parser.add_argument('--comet_workspace', type=str, required=True)
    parser.add_argument('-t', '--task', default='integral', choices={'integral', 'tail'})
    args = parser.parse_args()

    if args.mode == 'train':
        main_train(args)
    elif args.mode == 'eval':
        main_eval(args)
    else:
        raise ValueError
