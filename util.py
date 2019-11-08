import logging
import random

import numpy as np
import torch


def save_model(save_dir, generator, discriminator, generator_opt, discriminator_opt, epoch):
    save_dir.mkdir(exist_ok=True, parents=True)
    model_path = save_dir / f'model-{epoch}'

    logging.info(f'save model to {str(model_path)}')

    with model_path.open('wb') as f:
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_opt': generator_opt.state_dict(),
            'discriminator_opt': discriminator_opt.state_dict()
        }, f)


def load_model(model_path, generator, discriminator, generator_opt, discriminator_opt, device):
    logging.info(f'load model from {str(model_path)}')

    with model_path.open('rb') as f:
        weights = torch.load(f, map_location=device)
        generator.load_state_dict(weights['generator'])
        discriminator.load_state_dict(weights['discriminator'])
        if generator_opt is not None:
            generator_opt.load_state_dict(weights['generator_opt'])
        if discriminator_opt is not None:
            discriminator_opt.load_state_dict(weights['discriminator_opt'])


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
