import logging
import math

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from evaluation import evaluate_model
from util import save_model


class GANTrainer:
    def __init__(self, generator, discriminator, device='cpu', **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.kwargs = kwargs

    def train(self, parameters, train_dataset, optimizer_g, optimizer_d,
              experiment, scaler, save_dir,
              test_dataset=None, scheduler_d=None, scheduler_g=None):

        logging.info(f'Train for {parameters.iterations} iterations with BATCH_SIZE={parameters.batch_size} and '
                     f'TRAINING_RATIO={parameters.training_ratio}')

        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=parameters.batch_size)

        epochs_num = int(math.ceil(parameters.iterations / len(train_loader)))
        iterations_total = 0

        for epoch in range(epochs_num):
            for batch_num, X_batch_real in enumerate(tqdm(train_loader, desc=f'epoch {epoch}', position=0, leave=True)):
                self.generator.train()
                batch_size = X_batch_real.size(0)

                if (iterations_total + 1) % parameters.save_every == 0:
                    save_model(save_dir, self.generator, self.discriminator, optimizer_g, optimizer_d, iterations_total)
                if iterations_total > parameters.iterations:
                    break

                iterations_total += 1

                for _ in range(parameters.training_ratio):
                    d_loss = self.discriminator_loss(parameters, X_batch_real)

                    optimizer_d.zero_grad()
                    d_loss.backward()
                    optimizer_d.step()

                    if scheduler_d is not None:
                        scheduler_d.step()

                g_loss = self.generator_loss((batch_size, parameters.gan_noise_size))

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

                if scheduler_g is not None:
                    scheduler_g.step()

                if experiment is not None and iterations_total % parameters.log_every == 0:
                    assert test_dataset is not None
                    experiment.log_metrics({'g_loss': g_loss.detach().cpu().numpy(),
                                            'd_loss': d_loss.detach().cpu().numpy()})
                    eval_batch_num = int((parameters.gan_test_ratio * len(test_dataset)) / parameters.eval_batch_size)
                    evaluate_model(self.generator, experiment,
                                   test_dataset, parameters.eval_batch_size,
                                   eval_batch_num, parameters,
                                   self.device, scaler, iterations_total)

        return iterations_total

    def discriminator_loss(self, parameters, X_batch_real):
        X_batch_real = X_batch_real.to(self.device)

        X_noise = torch.randn((X_batch_real.size(0), parameters.gan_noise_size)).to(self.device)
        G_output = self.generator(X_noise)

        D_fake = self.discriminator(G_output.float())
        D_real = self.discriminator(X_batch_real.float())

        D_loss = -torch.mean(torch.cat((torch.log(D_real + 1e-8),
                                        torch.log(1 - D_fake + 1e-8))))

        return D_loss

    def generator_loss(self, noise_batch_shape):
        X_noise = torch.randn(noise_batch_shape).to(self.device)
        G_output = self.generator(X_noise)
        D_output = self.discriminator(G_output.float())

        G_loss = -torch.mean(torch.log(D_output + 1e-8))

        return G_loss


class WGPGANTrainer(GANTrainer):
    def discriminator_loss(self, parameters, X_batch_real):
        X_batch_real = X_batch_real.to(self.device)

        X_noise = torch.randn((parameters.batch_size, parameters.gan_noise_size)).to(self.device)
        G_output = self.generator(X_noise).detach()

        D_fake = self.discriminator(G_output.float())
        D_real = self.discriminator(X_batch_real.float())

        epsilon = torch.rand(X_batch_real.shape[0], 1).expand(X_batch_real.size()).to(self.device)

        G_interpolation = epsilon * X_batch_real.float() + (1 - epsilon) * G_output.float()
        G_interpolation = torch.autograd.Variable(G_interpolation, requires_grad=True)
        D_interpolation = self.discriminator(G_interpolation)

        weight = torch.ones(D_interpolation.size(), device=self.device)

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        grad_penalty = self.kwargs['lambda_'] * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

        D_loss = torch.mean(D_fake) - torch.mean(D_real) + grad_penalty

        return D_loss

    def generator_loss(self, noise_size):
        X_noise = torch.randn(noise_size).to(self.device)
        G_output = self.generator(X_noise)
        D_output = self.discriminator(G_output.float())

        G_loss = -1 * (torch.mean(D_output))

        return G_loss
