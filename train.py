import logging

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from metrics import MetricsAccum


def train(generator, discriminator, parameters, train_dataset, optimizer_g, optimizer_d, device, experiment,
          scheduler_d=None, scheduler_g=None, criterion=nn.BCELoss()):
    logging.info(f'Train for {parameters.epochs} epochs with BATCH_SIZE={parameters.batch_size} and '
                 f'TRAINING_RATIO={parameters.training_ratio}')

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=parameters.batch_size)

    y_real = torch.ones((parameters.batch_size, 1), device=device)
    y_fake = torch.zeros((parameters.batch_size, 1), device=device)

    for epoch in tqdm(range(parameters.epochs), desc=f'training'):
        metric_accum = MetricsAccum()
        for batch_num, X_batch_real in enumerate(train_loader):
            if batch_num >= parameters.training_ratio:
                break
            X_batch_real = X_batch_real.to(device)

            X_noise = torch.randn((parameters.batch_size, parameters.gan_noise_size)).to(device)
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

        X_noise = torch.randn((parameters.batch_size, parameters.gan_noise_size)).to(device)

        predict = discriminator(generator(X_noise.cuda()))
        loss = criterion(predict, y_real)

        metric_accum.gen_accum(loss.item())

        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        metrics = metric_accum.calculate()
        if experiment is not None:  # this can be a bottleneck with small batch size, so it can be turned off
            experiment.log_metrics(vars(metrics), epoch=epoch)

        if scheduler_g is not None:
            scheduler_g.step()
