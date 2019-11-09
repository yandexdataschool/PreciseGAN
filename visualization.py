from matplotlib import pyplot as plt


def visualize_jet_feature_distribution(ax, inverse_generated, test_set, hist_bins, hist_range, title, bin_widths_t,
                                       chi2, paper_chi):
    ax.hist(inverse_generated, bins=hist_bins, range=hist_range, density=True)
    ax.hist(test_set, bins=hist_bins, range=hist_range, alpha=0.5, density=True)
    ax.set_title(title)
    ax.set_ylabel('Events / Bin Width')

    plt.xlim(min(bin_widths_t), max(bin_widths_t))

    plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2, 1)}', horizontalalignment='right',
             verticalalignment='top', transform=ax.transAxes)

    plt.text(0.9, 0.8, f'paper χ2/NDF: {paper_chi}', horizontalalignment='right', fontdict={'color': 'red'},
             verticalalignment='top', transform=ax.transAxes)


def visualize_dijet_system(jj_M_gan, jj_M_test, n_bins_chi, range_chi, chi2_tail, article_chi_tail, ax, fig_tail_chi,
                           experiment):
    ax[0].set_title('Linear hist dijet system m')
    ax[0].hist(jj_M_gan[:, 2], bins=n_bins_chi, range=range_chi, density = True)
    ax[0].hist(jj_M_test[:, 2], bins=n_bins_chi, range=range_chi, alpha=0.5, density = True)
    ax[1].set_title('Log hist dijet system m')
    ax[1].hist(jj_M_gan[:, 2], bins=n_bins_chi, range=range_chi, log=True, density=True)
    ax[1].hist(jj_M_test[:, 2], bins=n_bins_chi, range=range_chi, log=True, alpha=0.5, density=True)

    plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
             verticalalignment='top', transform=ax[0].transAxes)

    plt.text(0.9, 0.875, f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right',
             fontdict={'color': 'red'},
             verticalalignment='top', transform=ax[0].transAxes)

    plt.text(0.9, 0.9, f'χ2/NDF: {round(chi2_tail, 1)}', horizontalalignment='right',
             verticalalignment='top', transform=ax[1].transAxes)

    plt.text(0.9, 0.875, f'paper χ2/NDF: {article_chi_tail}', horizontalalignment='right',
             fontdict={'color': 'red'},
             verticalalignment='top', transform=ax[1].transAxes)

    fig_tail_chi.show()
    experiment.log_figure(figure_name='fig_tail_chi_M_system_distribution', figure=fig_tail_chi)