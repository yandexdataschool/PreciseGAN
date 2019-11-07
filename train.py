import logging
import math

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from evaluation import evaluate_model
from metrics import MetricsAccum
from util import save_model


def train(generator, discriminator, parameters, train_dataset, optimizer_g, optimizer_d, device, experiment, scaler,
          save_dir, test_dataset=None, scheduler_d=None, scheduler_g=None, criterion=nn.BCELoss(), plot_tail = False):
    logging.info(f'Train for {parameters.iterations} iterations with BATCH_SIZE={parameters.batch_size} and '
                 f'TRAINING_RATIO={parameters.training_ratio}')

    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters.batch_size)

    epochs_num = int(math.ceil(parameters.iterations / len(train_loader)))
    iterations_total = 0

    for epoch in range(epochs_num):
        metric_accum = MetricsAccum()
        for batch_num, X_batch_real in enumerate(tqdm(train_loader, desc=f'epoch {epoch}', position=0, leave=True)):
            batch_size = X_batch_real.size(0)
            y_real = torch.ones((batch_size, 1), device=device)
            y_fake = torch.zeros((batch_size, 1), device=device)
            if (iterations_total + 1) % parameters.save_every == 0:
                save_model(save_dir, generator, discriminator, optimizer_g, optimizer_d, iterations_total)
            if iterations_total > parameters.iterations:
                break
            iterations_total += 1
            if batch_num % parameters.training_ratio == 0:
                X_noise = torch.randn((batch_size, parameters.gan_noise_size), device=device)

                predict = discriminator(generator(X_noise.cuda()))
                loss = criterion(predict, y_real)

                metric_accum.gen_accum(loss.item())

                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

                if scheduler_g is not None:
                    scheduler_g.step()

            X_batch_real = X_batch_real.to(device)

            X_noise = torch.randn((parameters.batch_size, parameters.gan_noise_size), device=device)
            X_batch_fake = generator(X_noise).detach()

            predict = discriminator(X_batch_real.float())
            loss = criterion(predict, y_real)

            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

            metric_accum.disc_accum(loss.item(), torch.round(predict), y_real)

            predict = discriminator(X_batch_fake)
            loss = criterion(predict, y_fake)

            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

            metric_accum.disc_accum(loss.item(), torch.round(predict), y_fake)

            if scheduler_d is not None:
                scheduler_d.step()

            if experiment is not None and iterations_total % parameters.log_every == 0:
                assert test_dataset is not None
                metrics = metric_accum.calculate()
                experiment.log_metrics(vars(metrics), epoch=epoch)
                eval_batch_num = len(test_dataset) // parameters.eval_batch_size
                evaluate_model(generator, experiment, test_dataset, parameters.eval_batch_size, eval_batch_num, parameters,
                               device, scaler, iterations_total, plot_tail)

    return iterations_total
