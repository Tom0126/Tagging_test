import numpy as np
import matplotlib.pyplot as plt


def calculate_auc(tpr, fpr):
    auc = 0

    tpr = tpr[fpr > 0]
    fpr = fpr[fpr > 0]

    n = len(tpr)
    for i in range(n - 1):
        auc += (tpr[i] + tpr[i + 1]) * (fpr[i] - fpr[i + 1]) / 2
    return auc


def plotROC(fpr_path, tpr_path, auroc_path, save_path, **kwargs):
    fprs = np.load(fpr_path, allow_pickle=True)
    tprs = np.load(tpr_path, allow_pickle=True)
    auroc = np.load(auroc_path, allow_pickle=True)

    fpr = fprs[kwargs.get('dim', 1)]
    tpr = tprs[kwargs.get('dim', 1)]
    auc = auroc[kwargs.get('dim', 1)]

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    auc_tag = kwargs.get('tag', 'ANN') + ' AUC: {:.3f}'.format(auc)
    plt.plot(tpr, 1 - fpr, label=auc_tag, color='red')

    plt.xlabel('Signal efficiency', fontsize=10)
    plt.ylabel('Background rejection rate', fontsize=10)

    plt.ylim(0, 1.2)
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))

    plt.legend(bbox_to_anchor=(0.9, 0.9), bbox_transform=ax.transAxes, fontsize=13, loc='upper right')
    plt.savefig(save_path)
    plt.close(fig)

    # plt.show()


def plot_s_b_threshold(fpr_path, tpr_path, save_path, threshold_num, **kwargs):
    fprs = np.load(fpr_path, allow_pickle=True)
    tprs = np.load(tpr_path, allow_pickle=True)

    fpr = fprs[kwargs.get('dim', 1)]
    tpr = tprs[kwargs.get('dim', 1)]
    bkr = 1 - fpr

    thresholds = np.linspace(1, 0, threshold_num)

    assert len(tpr) == threshold_num

    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_subplot(111)

    plt.text(0.1, 0.9, kwargs.get('tag', ' '), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    l1 = ax.plot(thresholds[::5], tpr[::5], 'o', label='Signal', color='red', markersize=6)
    ax2 = ax.twinx()
    l2 = ax2.plot(thresholds[::5], bkr[::5], '^', label='Background', color='black', markersize=6)

    ax.set_xlabel('ANN probability threshold', fontsize=14)
    ax.set_ylabel('Signal efficiency' + r'$(N_{S}^{sel.}/N_{S})$', fontsize=14)
    ax2.set_ylabel('Bkg. rejection rate ' + r'$(1- N_{B}^{sel.}/N_{B}$)', fontsize=14)

    ax.tick_params(labelsize=14, direction='in', length=5)
    ax2.tick_params(labelsize=14, direction='in', length=5)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0.9, 1, 6))
    ax2.set_yticks(np.linspace(0.9, 1, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.02)
    ax2.set_ylim(0.9, 1.02)

    # plt.minorticks_on()
    #
    # ax.tick_params(which='minor', direction='in', length=3)
    # ax2.tick_params(which='minor', direction='in', length=3)
    #
    # ax.set_xticks(np.linspace(0, 1, 51), minor=True)
    # ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)
    # ax2.set_yticks(np.linspace(0.9, 1, 26), minor=True)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.9, 0.98), fontsize=14)

    plt.savefig(save_path)
    plt.close(fig)


def plot_s_b_ratio_threshold(fpr_path, tpr_path, save_path, threshold_num, **kwargs):
    label_size = 18

    fprs = np.load(fpr_path, allow_pickle=True)
    tprs = np.load(tpr_path, allow_pickle=True)

    fpr = fprs[kwargs.get('dim', 1)]
    tpr = tprs[kwargs.get('dim', 1)]
    bkr = 1 / fpr

    thresholds = np.linspace(1, 0, threshold_num)

    assert len(tpr) == threshold_num

    fig = plt.figure(figsize=(8, 7))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(111)

    plt.text(0.1, 0.9, kwargs.get('tag', ' '), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    l1 = ax.plot(thresholds[::5], tpr[::5], 'o', label='Signal', color='red', markersize=6)
    ax2 = ax.twinx()
    l2 = ax2.plot(thresholds[::5], bkr[::5], '^', label='Background', color='black', markersize=6)

    ax.set_xlabel('Likelihood threshold', fontsize=label_size)
    ax.set_ylabel('Signal efficiency' + r'$(N_{S}^{sel.}/N_{S})$', fontsize=label_size - 2)
    ax2.set_ylabel('Bkg. rejection ' + r'$N_{B}/(N_{B}^{sel.}$)', fontsize=label_size - 2)

    ax.tick_params(labelsize=label_size - 2, direction='in', length=5)
    ax2.tick_params(labelsize=label_size - 2, direction='in', length=5)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0.9, 1, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.03)
    ax2.set_ylim(1, 10 * np.amax(bkr[::5][~np.isinf(bkr[::5])]))

    # plt.minorticks_on()
    #
    # ax.tick_params(which='minor', direction='in', length=3)
    # ax2.tick_params(which='minor', direction='in', length=3)
    #
    # ax.set_xticks(np.linspace(0, 1, 51), minor=True)
    # ax.set_yticks(np.linspace(0.9, 1, 26), minor=True)

    ax2.set_yscale('log')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(0.95, 0.98), fontsize=label_size - 2)

    plt.savefig(save_path)
    plt.close(fig)


def plot_s_b_ep(threshold, tpr_file_lists, fpr_file_lists, ep_lists, signal, save_path, threshold_num):
    particle_dim = {'mu+': 0, 'e+': 1, 'pi+': 2, 'noise': 3}
    particle_name = {'mu+': r'$\mu^+$', 'e+': r'$e^+$', 'pi+': r'$\pi^+$', 'noise': 'Noise'}

    tpr = []
    bkr = []

    assert len(ep_lists) == len(tpr_file_lists)
    assert len(ep_lists) == len(fpr_file_lists)

    for tpr_path, fpr_path in zip(tpr_file_lists, fpr_file_lists):
        tpr_ = np.load(tpr_path)
        fpr_ = np.load(fpr_path)

        tpr.append(tpr_[particle_dim.get(signal), -1 * int(threshold * (threshold_num))])
        bkr.append(1 - fpr_[particle_dim.get(signal), -1 * int(threshold * (threshold_num))])
        tpr_ = None
        fkr_ = None

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    l1 = ax.plot(ep_lists, tpr, 'o', label=particle_name.get(signal), color='red')
    ax2 = ax.twinx()
    l2 = ax2.plot(ep_lists, bkr, '^', label='Backgrounds', color='black')

    ax.set_xlabel('Energy [GeV]', fontsize=10)
    ax.set_ylabel('{} efficiency'.format(particle_name.get(signal)), fontsize=10)
    ax2.set_ylabel('Background rejection rate', fontsize=10)

    plt.text(0.1, 0.9, 'CEPC AHCAL', fontsize=15, fontstyle='oblique', fontweight='bold', horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )
    plt.text(0.1, 0.84, 'AHCAL PID Threshold = {}'.format(threshold), fontsize=12, fontstyle='normal',
             horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, )

    ax.set_xticks(ep_lists)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax2.set_yticks(np.linspace(0, 1, 11))

    ax.set_ylim(0, 1.3)
    ax2.set_ylim(0, 1.3)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right')

    # plt.legend(bbox_to_anchor=(0.1, 66),bbox_transform=ax.transAxes)
    plt.savefig(save_path.format(signal))
    plt.close(fig)


if __name__ == '__main__':
    fpr_path = '../roc/fpr.npy'
    tpr_path = '../roc/tpr.npy'
    auroc_path = '../roc/auroc.npy'
    bdt_path = '../roc/pion_roc_bdt.txt'
    save_path = 'Fig/ann_bdt_compare.png'
    plotROC(fpr_path=fpr_path, tpr_path=tpr_path, auroc_path=auroc_path, signal='pi+', save_path=save_path)
