import numpy as np
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import PTCL_FEATURES, DIJET_SYSTEM_FEATURES

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

    hist_bins = [20, 25, 30, 20, 25, 30, 15, 30, 20]
    hist_ranges = [(200, 800), (-2.5, 2.5), (0, 300), (200, 600), (-2.5, 2.5),
                   (0, 300), (0, 300), (-6, 6), (0, 2000)]
    start_bin = [1, 0, 0, 2, 0, 0, 0, 0, 5]
    if parametres.level == 'ptcl':
        paper_chi = [794.7, 86.7, 525.8, 1010.8, 21.6, 1248.1, 855.5, 104.2, 906.9]
    else:
        paper_chi =[164.9, 200.8, 2467.9, 1388.7, 174.3, 485.1, 1849.7, 1009.0, 76.9]
    chisqs = []
    ks_tests = []

    for i in range(inverse_generated.shape[1]):
        count_g, bin_widths_g = np.histogram(inverse_generated[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])
        count_t, bin_widths_t = np.histogram(test_set[:, i],
                                             bins=hist_bins[i], range=hist_ranges[i])


        ax[i // 3][i % 3].hist(inverse_generated[:, i], bins=hist_bins[i], range=hist_ranges[i], density = True)
        ax[i // 3][i % 3].hist(test_set[:, i], bins=hist_bins[i], range=hist_ranges[i], alpha=0.5, density = True)
        ax[i // 3][i % 3].set_title(features[i])
        ax[i // 3][i % 3].set_ylabel('Events / Bin Width')

        plt.xlim(min(bin_widths_t), max(bin_widths_t))

        chi2 = stats.chisquare(count_t[start_bin[i]:], (count_g)[start_bin[i]:])[0] / (hist_bins[i]- start_bin[i] - 1)
        chisqs.append(chi2)

        ks = stats.ks_2samp(inverse_generated[:, i], test_set[:, i])
        ks_tests.append(ks)

        plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2, 1)}', horizontalalignment='right',
                 verticalalignment='top', transform=ax[i // 3][i % 3].transAxes)

        plt.text(0.9, 0.8, f'paper χ2/NDF: {paper_chi[i]}', horizontalalignment='right', fontdict={'color': 'red'},
                 verticalalignment='top', transform=ax[i // 3][i % 3].transAxes)

    fig.title = f'architecture: {parametres.architecture}'

    experiment.log_figure(figure_name='distributions', figure=fig, step=step)
    fig.clear()
    plt.close(fig)
    experiment.log_metrics({f'chi2_st_f{i}': chisq for i, chisq in enumerate(chisqs)}, step=step)

    experiment.log_metrics({f'ks_st_f{i}': ks for i, (ks, pval) in enumerate(ks_tests)}, step=step)
    experiment.log_metrics({f'ks_pval_f{i}': pval for i, (chisq, pval) in enumerate(ks_tests)}, step=step)

    if (parametres.task == 'tail'):
         fig_tail_chi, ax = plt.subplots(1, 2, figsize=(20, 8))
         n_bins_chi = 55
         article_chi_tail = 1.0
         chisqs_tail = []
         range_chi = [2500, 8000]
         count_g, bin_widths_g = np.histogram(jj_M_gan[:,2], bins = n_bins_chi, range = range_chi)
         count_t, bin_widths_t = np.histogram(jj_M_test[:, 2], bins = n_bins_chi, range = range_chi)

         chi2_tail = stats.chisquare(count_t, count_g)[0] / (n_bins_chi - 1)
         chisqs_tail.append(chi2_tail)

         ax[0].set_title('Linear hist didjet system m')
         ax[0].hist(jj_M_gan[:,2], bins=n_bins_chi, range= range_chi)
         ax[0].hist(jj_M_test[:,2], bins=n_bins_chi, range= range_chi, alpha=0.5)
         ax[1].set_title('Log hist didjet system m')
         ax[1].hist(jj_M_gan[:,2], bins=n_bins_chi, range=range_chi, log=True)
         ax[1].hist(jj_M_test[:,2], bins=n_bins_chi, range=range_chi, log=True, alpha=0.5)

         plt.text(0.9, 0.9,  f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
                  verticalalignment='top', transform=ax[0].transAxes)

         plt.text(0.9, 0.875, f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right', fontdict={'color': 'red'},
                  verticalalignment='top', transform=ax[0].transAxes)

         plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
                  verticalalignment='top', transform=ax[1].transAxes)

         plt.text(0.9, 0.875, f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right', fontdict={'color': 'red'},
                  verticalalignment='top', transform=ax[1].transAxes)

         fig_tail_chi.show()
         experiment.log_figure(figure_name='fig_tail_chi_M_system_distribution', figure=fig_tail_chi)

         fig_tail, ax = plt.subplots(1, 2, figsize=(20, 8))
         n_bins = 70
         range_hist = [1000, 8000]

         ax[0].set_title('Linear hist didjet system m')
         ax[0].hist(jj_M_gan[:,2], bins=n_bins, range= range_hist)
         ax[0].hist(jj_M_test[:,2], bins=n_bins, range= range_hist, alpha=0.5)
         ax[1].set_title('Log hist didjet system m')
         ax[1].hist(jj_M_gan[:,2], bins=n_bins, range=range_hist, log=True)
         ax[1].hist(jj_M_test[:,2], bins=n_bins, range=range_hist, log=True, alpha=0.5)

         plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
                  verticalalignment='top', transform=ax[0].transAxes)

         plt.text(0.9, 0.875,  f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right', fontdict={'color': 'red'},
                  verticalalignment='top', transform=ax[0].transAxes)

         plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
                  verticalalignment='top', transform=ax[1].transAxes)

         plt.text(0.9, 0.875,  f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right', fontdict={'color': 'red'},
                  verticalalignment='top', transform=ax[1].transAxes)

         fig_tail.show()
         experiment.log_figure(figure_name='tail_M_system_distribution', figure=fig_tail)



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