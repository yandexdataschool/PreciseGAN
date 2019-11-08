import numpy as np
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import PTCL_FEATURES, DIJET_SYSTEM_FEATURES
from paper_invariants import PAPER_HIST_BINS, PAPER_HIST_RANGES, PAPER_START_BINS, PAPER_PTCL_CHI2_STATISTICS, \
    PAPER_CHI2_TAIL_STATISTICS
from paper_invariants import PAPER_RECO_CHI2_STATISTICS
from visualization import visualize_jet_feature_distribution, visualize_dijet_system

ANGLE_IDX = 5


def evaluate_model(generator, experiment, test_set, batch_size, batch_num, parametres, device, scaler, step):
    features = PTCL_FEATURES.copy()
    del features[ANGLE_IDX]
    features += DIJET_SYSTEM_FEATURES
    predictions = []
    for _ in tqdm(range(batch_num), desc='evaluation', position=0, leave=True):
        x_noise = torch.randn((batch_size, parametres.gan_noise_size), device=device)
        predictions.append(generator(x_noise).cpu().detach().numpy())

    predictions_np = np.concatenate(predictions)

    inverse_generated = scaler.inverse_transform(predictions_np)

    jj_M_gan = compute_jj(inverse_generated)
    jj_M_test = compute_jj(test_set)

    inverse_generated = np.delete(inverse_generated, ANGLE_IDX, axis=1)
    test_set = np.delete(test_set, ANGLE_IDX, axis=1)

    inverse_generated = np.concatenate((inverse_generated, jj_M_gan), 1)
    test_set = np.concatenate((test_set, jj_M_test), 1)

    fig, ax = plt.subplots(3, 3, figsize=(20, 12))

    if parametres.level == 'ptcl':
        paper_chi = PAPER_PTCL_CHI2_STATISTICS
    else:
        paper_chi = PAPER_RECO_CHI2_STATISTICS
    chisqs = []
    ks_tests = []

    for i, feature_name in enumerate(features):
        count_g, bin_widths_g = np.histogram(inverse_generated[:, i], bins=PAPER_HIST_BINS[i],
                                             range=PAPER_HIST_RANGES[i])
        count_t, bin_widths_t = np.histogram(test_set[:, i], bins=PAPER_HIST_BINS[i], range=PAPER_HIST_RANGES[i])

        chi2 = stats.chisquare(count_t[PAPER_START_BINS[i]:], count_g[PAPER_START_BINS[i]:])[0]
        chi2 /= (PAPER_HIST_BINS[i] - PAPER_START_BINS[i] - 1)
        chisqs.append(chi2)

        ks = stats.ks_2samp(inverse_generated[:, i], test_set[:, i])
        ks_tests.append(ks)

        visualize_jet_feature_distribution(ax[i // 3][i % 3], inverse_generated[:, i], test_set[:, i],
                                           PAPER_HIST_BINS[i], PAPER_HIST_RANGES[i], feature_name, bin_widths_t, chi2,
                                           paper_chi[i])

    fig.suptitle(f'architecture: {parametres.architecture}\nComparison of kinematic observables with respect to '
                 f'{parametres.level}-level Monte Carlo simulation')

    experiment.log_figure(figure_name='distributions', figure=fig, step=step)

    fig.clear()
    plt.close(fig)

    experiment.log_metrics({f'chi2_st_f{i}': chisq for i, chisq in enumerate(chisqs)}, step=step)
    experiment.log_metrics({f'ks_st_f{i}': ks for i, (ks, pval) in enumerate(ks_tests)}, step=step)
    experiment.log_metrics({f'ks_pval_f{i}': pval for i, (chisq, pval) in enumerate(ks_tests)}, step=step)

    if parametres.task == 'tail':
        fig_tail_chi, ax = plt.subplots(1, 2, figsize=(20, 8))
        n_bins_chi = 55
        chisqs_tail = []
        range_chi = [2500, 8000]
        count_g, bin_widths_g = np.histogram(jj_M_gan[:, 2], bins=n_bins_chi, range=range_chi)
        count_t, bin_widths_t = np.histogram(jj_M_test[:, 2], bins=n_bins_chi, range=range_chi)

        chi2_tail = stats.chisquare(count_t, count_g)[0] / (n_bins_chi - 1)
        chisqs_tail.append(chi2_tail)

        visualize_dijet_system(jj_M_gan, jj_M_test, n_bins_chi, range_chi, chi2_tail, PAPER_CHI2_TAIL_STATISTICS, ax,
                               fig_tail_chi, experiment)

        fig_tail, ax = plt.subplots(1, 2, figsize=(20, 8))
        n_bins = 70
        range_hist = [1000, 8000]

        visualize_dijet_system(jj_M_gan, jj_M_test, n_bins, range_hist, chi2_tail, PAPER_CHI2_TAIL_STATISTICS, ax,
                               fig_tail, experiment)


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

    return np.column_stack((jj_pt, jj_eta, jj_M))
