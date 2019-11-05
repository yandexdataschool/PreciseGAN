import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm


def evaluate_model(generator, experiment, test_set, batch_size, batch_num, gan_noise_size, device):
    predictions = []
    for _ in tqdm(range(batch_num), desc='evaluation'):
        x_noise = torch.randn((batch_size, gan_noise_size)).to(device)
        predictions.append(generator(x_noise).cpu().detach().numpy())

    predictions_np = np.concatenate(predictions)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(7):
        sns.kdeplot(predictions_np[:, i].T, color='red', alpha=0.7, label='generated', ax=ax[i // 4][i % 4])
        sns.kdeplot(test_set[:, i].T, color='blue', alpha=0.7, label='test', ax=ax[i // 4][i % 4])
    fig.legend()

    experiment.log_figure(fig)
