import numpy as np
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from data import PTCL_FEATURES

ANGLE_IDX = 5


def evaluate_model(generator, experiment, test_set, batch_size, batch_num, parametres, device, scaler):
    features = PTCL_FEATURES.copy()
    del features[ANGLE_IDX]
    predictions = []
    for _ in tqdm(range(batch_num), desc='evaluation', position=0, leave=True):
        x_noise = torch.randn((batch_size, parametres.gan_noise_size)).to(device)
        predictions.append(generator(x_noise).cpu().detach().numpy())

    predictions_np = np.concatenate(predictions)

    inverse_generated = scaler.inverse_transform(predictions_np)

    inverse_generated = np.delete(inverse_generated, ANGLE_IDX, axis=1)
    test_set = np.delete(test_set, ANGLE_IDX, axis=1)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))

    hist_bins = [20, 25, 30, 20, 25, 30, ]
    hist_ranges = [(200, 800), (-2.5, 2.5), (0, 300), (200, 600), (-2.5, 2.5),
                   (0, 300)]
    start_bin = [1, 0, 0, 2, 0, 0]
    chisqs = []

    for i in range(inverse_generated.shape[1]):
        count_g, bin_widths_g = np.histogram(inverse_generated[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])
        count_t, bin_widths_t = np.histogram(test_set[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])

        width = (max(bin_widths_t) - min(bin_widths_t)) / hist_bins[i]

        ax[i // 4][i % 4].bar(bin_widths_g[:-1], count_g, width=width)
        ax[i // 4][i % 4].label = features[i]
        ax[i // 4][i % 4].bar(bin_widths_t[:-1], count_t, width=width, alpha=0.5)

        plt.xlim(min(bin_widths_t), max(bin_widths_t))

        chi2 = stats.chisquare(count_t[start_bin[i]:], (count_g)[start_bin[i]:])
        chisqs.append(chi2)

    fig.title = f'architecture: {parametres.architecture}'
    fig.show()

    # FIXME: not working
    # experiment.log_figure(figure_name='distributions', figure=fig)
    experiment.log_metrics({f'chisq{i}': chisq for i, (chisq, pval) in enumerate(chisqs)})
    experiment.log_metrics({f'pval{i}': pval for i, (chisq, pval) in enumerate(chisqs)})
