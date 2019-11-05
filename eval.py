import numpy as np
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def evaluate_model(generator, experiment, test_set, batch_size, batch_num, gan_noise_size, device, scaler):
    predictions = []
    for _ in tqdm(range(batch_num), desc='evaluation', position=0, leave=True):
        x_noise = torch.randn((batch_size, gan_noise_size)).to(device)
        predictions.append(generator(x_noise).cpu().detach().numpy())

    predictions_np = np.concatenate(predictions)

    inverse_generated = scaler.inverse_transform(predictions_np)
    test_inverse = scaler.inverse_transform(test_set)

    inverse_generated = inverse_generated[:, [0, 1, 2, 3, 4, 6]]
    test_inverse = test_inverse[:, [0, 1, 2, 3, 4, 6]]

    fig, ax = plt.subplots(2, 4, figsize=(20, 12))
    hist_bins = [20, 25, 30, 20, 25, 30, ]
    hist_ranges = [(200, 800), (-2.5, 2.5), (0, 300), (200, 600), (-2.5, 2.5),
                   (0, 300)]
    start_bin = [1, 0, 0, 2, 0, 0]
    chisqs = []

    for i in range(6):
        count_g, bin_widths_g = np.histogram(inverse_generated[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])
        count_t, bin_widths_t = np.histogram(test_inverse[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])

        width = (max(bin_widths_t) - min(bin_widths_t)) / hist_bins[i]

        ax[i // 4][i % 4].bar(bin_widths_g[:-1], count_g, width=width)
        ax[i // 4][i % 4].bar(bin_widths_t[:-1], count_t, width=width, alpha=0.5)

        plt.xlim(min(bin_widths_t), max(bin_widths_t))

        chi2 = stats.chisquare(count_t[start_bin[i]:], (count_g)[start_bin[i]:])
        chisqs.append(chi2)

    fig.show()

    # FIXME: not working
    # experiment.log_figure(figure_name='distributions', figure=fig)
    experiment.log_metrics({f'chisq{i}': chisq for i, (chisq, pval) in enumerate(chisqs)})
    experiment.log_metrics({f'pval{i}': pval for i, (chisq, pval) in enumerate(chisqs)})
