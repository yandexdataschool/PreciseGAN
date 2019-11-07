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

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    hist_bins = [20, 25, 30, 20, 25, 30, ]
    hist_ranges = [(200, 800), (-2.5, 2.5), (0, 300), (200, 600), (-2.5, 2.5),
                   (0, 300)]
    start_bin = [1, 0, 0, 2, 0, 0]
    article_chi = [794.7, 86.7, 525.8, 1010.8, 21.6, 1248.1]
    chisqs = []
    ks_tests = []

    for i in range(inverse_generated.shape[1]):
        count_g, bin_widths_g = np.histogram(inverse_generated[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])
        count_t, bin_widths_t = np.histogram(test_set[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])

        width = (max(bin_widths_t) - min(bin_widths_t)) / hist_bins[i]

        ax[i // 3][i % 3].bar(bin_widths_g[:-1], count_g, width=width)
        ax[i // 3][i % 3].label = features[i]
        ax[i // 3][i % 3].bar(bin_widths_t[:-1], count_t, width=width, alpha=0.5)

        plt.xlim(min(bin_widths_t), max(bin_widths_t))

        chi2 = stats.chisquare(count_t[start_bin[i]:], (count_g)[start_bin[i]:])[0] / (hist_bins[i] - 1)
        chisqs.append(chi2)

        ks = stats.ks_2samp(inverse_generated[:, i], test_set[:, i])
        ks_tests.append(ks)

        plt.text(0.9, 0.9, round(chi2, 1), horizontalalignment='right',
                 verticalalignment='top', transform=ax[i // 3][i % 3].transAxes)

        plt.text(0.9, 0.85, article_chi[i], horizontalalignment='right', fontdict={'color': 'red'},
                 verticalalignment='top', transform=ax[i // 3][i % 3].transAxes)

    fig.title = f'architecture: {parametres.architecture}'
    fig.legend()
    fig.show()

    # FIXME: not working
    # experiment.log_figure(figure_name='distributions', figure=fig)
    experiment.log_metrics({f'chisq_st_f{i}': chisq for i, (chisq, pval) in enumerate(chisqs)})
    experiment.log_metrics({f'chisq_pval_f{i}': pval for i, (chisq, pval) in enumerate(chisqs)})

    experiment.log_metrics({f'ks_st_f{i}': ks for i, (ks, pval) in enumerate(ks_tests)})
    experiment.log_metrics({f'ks_pval_f{i}': pval for i, (chisq, pval) in enumerate(ks_tests)})


def compute_jj(predictions):
    pt_1 = predictions[:, 0]
    eta_1 = predictions[:, 1]
    M_1 = predictions[:, 2]
    pt_2 = predictions[:, 3]
    eta_2 = predictions[:, 4]
    phi = predictions[:, 5]
    M_2 = predictions[:, 6]

    x1 = pt_1
    y1 = 0
    z1 = pt_1 * np.sinh(eta_1)
    t1 = np.sqrt(np.square(x1) + np.square(y1) + np.square(z1) + np.square(M_1))

    x2 = pt_2 * np.cos(phi)
    y2 = pt_2 * np.sin(phi)
    z2 = pt_2 * np.sinh(eta_2)
    t2 = np.sqrt(np.square(x2) + np.square(y2) + np.square(z2) + np.square(M_2))

    x3 = x1 + x2
    y3 = y1 + y2
    z3 = z1 + z2
    t3 = t1 + t2

    jj_M = np.sqrt(np.square(t3) - (np.square(x3) + np.square(y3) + np.square(z3)))
    jj_pt = np.sqrt(np.square(x3) + np.square(y3))
    jj_eta = np.arcsinh(z3 / jj_pt)

    return np.array((jj_pt, jj_eta, jj_M))