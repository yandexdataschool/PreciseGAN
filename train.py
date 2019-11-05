import logging

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from eval import evaluate_model
from metrics import MetricsAccum


def train(generator, discriminator, parameters, train_dataset, optimizer_g, optimizer_d, device, experiment,
          test_dataset=None, ecaluate_every=None, scheduler_d=None, scheduler_g=None, criterion=nn.BCELoss()):
    logging.info(f'Train for {parameters.epochs} epochs with BATCH_SIZE={parameters.batch_size} and '
                 f'TRAINING_RATIO={parameters.training_ratio}')

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=parameters.batch_size)

    y_real = torch.ones((parameters.batch_size, 1), device=device)
    y_fake = torch.zeros((parameters.batch_size, 1), device=device)

    for epoch in tqdm(range(parameters.epochs), desc=f'training', position=0, leave=True):
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

        if scheduler_g is not None:
            scheduler_g.step()

        if ecaluate_every is not None and experiment is not None:
            assert test_dataset is not None
            if (1 + epoch) % ecaluate_every == 0:
                metrics = metric_accum.calculate()
                experiment.log_metrics(vars(metrics), epoch=epoch)
                eval_batch_size = 512
                eval_batch_num = len(test_dataset) // 512
                evaluate_model(generator, experiment, test_dataset, eval_batch_size, eval_batch_num,
                               parameters.gan_noise_size, device)
