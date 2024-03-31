import numpy as np
import matplotlib.pyplot as plt
import os

# mu_path='../ahcal_bt_epoch_300_dis/mu+_dis.npy'
# e_path='../ahcal_bt_epoch_300_dis/e+_dis.npy'
# pi_path='../ahcal_bt_epoch_300_dis/pi+_dis.npy'
# log=True
# stack=False
# save_path='Fig/{}_dis{}{}.png'

def plotDistribution(mu_path, e_path, pi_path, log, stack, save_path, n_classes=3, proton_path=None):
    labels_dict = {3: ['mu+', 'e+', 'pion+'],
                   4: ['mu+', 'e+', 'pion+', 'proton']
                   }

    colors_dict = {3: ['green', 'blue', 'red'],
                   4: ['green', 'blue', 'red', 'orange']
                   }
    labels = labels_dict.get(n_classes)
    colors = colors_dict.get(n_classes)

    mu_dis = np.load(mu_path)
    mu_dis = np.transpose(mu_dis, (1, 0)) * 100  # row 0: mu+ pb, row 1: e+ pb, row 2: pi+ pb

    e_dis = np.load(e_path)
    e_dis = np.transpose(e_dis, (1, 0)) * 100

    pi_dis = np.load(pi_path)
    pi_dis = np.transpose(pi_dis, (1, 0)) * 100

    if n_classes == 4:
        proton_dis = np.load(proton_path)
        proton_dis = np.transpose(proton_dis, (1, 0)) * 100
    else:
        proton_dis = None

    hist_stype = 'barstacked'

    if log:
        log_tag = '_log'
    else:
        log_tag = ''
    #   mu+
    if stack:
        stack_tag = '_stack'
    else:
        stack_tag = ''

    fig = plt.figure(figsize=(6, 5))

    if log:
        plt.text(10, 0.28, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.14, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.07, 'Muon+ Events', fontsize=12, fontstyle='normal')
    else:
        plt.text(10, 0.9, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.84, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.78, 'Muon+ Events', fontsize=12, fontstyle='normal')

    if stack:
        plt.hist(np.transpose(mu_dis), bins=100, range=(0, 100), density=True
                 , histtype=hist_stype, label=labels, alpha=0.5, stacked=True,
                 color=colors, log=log)
    else:
        for i, particle in enumerate(labels):
            plt.hist(mu_dis[i], bins=100, range=(0, 100), density=True, color=colors[i]
                     , histtype=hist_stype, label=particle, alpha=0.5, log=log, stacked=stack)

    plt.xlabel('Probability [%]')
    plt.xticks(np.linspace(0, 100, 11))
    if not log:
        plt.ylim([0, 1.1])
    else:
        plt.ylim([0.00001, 1])
    plt.legend(loc='upper right')
    plt.savefig(save_path.format('mu', log_tag, stack_tag))
    # plt.show()
    plt.close(fig)

    ####################################
    #   e+

    fig = plt.figure(figsize=(6, 5))

    if log:
        plt.text(10, 0.28, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.14, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.07, 'Positron Events', fontsize=12, fontstyle='normal')
    else:
        plt.text(10, 0.9, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.84, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.78, 'Positron Events', fontsize=12, fontstyle='normal')

    if stack:
        plt.hist(np.transpose(e_dis), bins=100, range=(0, 100), density=True
                 , histtype=hist_stype, label=labels, alpha=0.5, stacked=True,
                 color=colors, log=log)
    else:
        for i, particle in enumerate(labels):
            plt.hist(e_dis[i], bins=100, range=(0, 100), density=True, color=colors[i]
                     , histtype=hist_stype, label=particle, alpha=0.5, log=log, stacked=stack)

    plt.xlabel('Probability [%]')
    plt.xticks(np.linspace(0, 100, 11))
    if not log:
        plt.ylim([0, 1.1])
    else:
        plt.ylim([0.00001, 1])
    plt.legend(loc='upper right')
    plt.savefig(save_path.format('e', log_tag, stack_tag))
    # plt.show()
    plt.close(fig)
    ####################################
    #   pion+

    fig = plt.figure(figsize=(6, 5))

    if log:
        plt.text(10, 0.28, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.14, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.07, 'Pion+ Events', fontsize=12, fontstyle='normal')
    else:
        plt.text(10, 0.9, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
        plt.text(10, 0.84, 'AHCAL PID', fontsize=12, fontstyle='normal')
        plt.text(10, 0.78, 'Pion+ Events', fontsize=12, fontstyle='normal')

    if stack:
        plt.hist(np.transpose(pi_dis), bins=100, range=(0, 100), density=True
                 , histtype=hist_stype, label=labels, alpha=0.5, stacked=True,
                 color=colors, log=log)
    else:
        for i, particle in enumerate(labels):
            plt.hist(pi_dis[i], bins=100, range=(0, 100), density=True, color=colors[i]
                     , histtype=hist_stype, label=particle, alpha=0.5, log=log, stacked=stack)
    plt.xlabel('Probability [%]')
    plt.xticks(np.linspace(0, 100, 11))
    if not log:
        plt.ylim([0, 1.1])
    else:
        plt.ylim([0.00001, 1])
    plt.legend(loc='upper right')
    plt.savefig(save_path.format('pi', log_tag, stack_tag))
    # plt.show()
    plt.close(fig)

    ####################################
    #   proton
    if n_classes==4:
        fig = plt.figure(figsize=(6, 5))

        if log:
            plt.text(10, 0.28, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
            plt.text(10, 0.14, 'AHCAL PID', fontsize=12, fontstyle='normal')
            plt.text(10, 0.07, 'Proton Events', fontsize=12, fontstyle='normal')
        else:
            plt.text(10, 0.9, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
            plt.text(10, 0.84, 'AHCAL PID', fontsize=12, fontstyle='normal')
            plt.text(10, 0.78, 'Proton Events', fontsize=12, fontstyle='normal')

        if stack:
            plt.hist(np.transpose(pi_dis), bins=100, range=(0, 100), density=True
                     , histtype=hist_stype, label=labels, alpha=0.5, stacked=True,
                     color=colors, log=log)
        else:
            for i, particle in enumerate(labels):
                plt.hist(proton_dis[i], bins=100, range=(0, 100), density=True, color=colors[i]
                         , histtype=hist_stype, label=particle, alpha=0.5, log=log, stacked=stack)
        plt.xlabel('Probability [%]')
        plt.xticks(np.linspace(0, 100, 11))
        if not log:
            plt.ylim([0, 1.1])
        else:
            plt.ylim([0.00001, 1])
        plt.legend(loc='upper right')
        plt.savefig(save_path.format('proton', log_tag, stack_tag))
        # plt.show()
        plt.close(fig)


def plot_ann_score(ana_dict, save_path,bins=100,**kwargs):


    label_col = kwargs.get('label_col', 'label')
    signal_col = kwargs.get('signal_col', 'signal')


    raw_labels=ana_dict[label_col]
    signal_score= ana_dict[signal_col]



    label_size=18


    fig=plt.figure(figsize=(8,7))
    ax = plt.gca()

    plt.text(0.05, 0.95, '', fontsize=label_size, fontstyle='oblique', fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes, )



    n_count=[]


    n,_,_=plt.hist(signal_score[raw_labels==1], bins=bins, label='Signal', range=[0, 1], histtype='step',
             color='red',
             density=False, stacked=False, alpha=1, log=True, linewidth=3,
             weights=np.ones(len(signal_score[raw_labels==1]))/len(signal_score[raw_labels==1]),
             hatch='//',
             )

    n_count.append(n)

    n, _, _ = plt.hist(signal_score[raw_labels == 0], bins=bins, label='Bkg.', range=[0, 1], histtype='step',
                       color='blue',
                       density=False, stacked=False, alpha=1, log=True, linewidth=3,
                       weights=np.ones(len(signal_score[raw_labels == 0])) / len(signal_score[raw_labels == 0]),
                       hatch='\\\\',
                       )

    n_count.append(n)



    plt.legend(loc='upper right', bbox_to_anchor=(0.92,0.98), fontsize=label_size)

    plt.xticks(np.linspace(0,1,11), fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.xlabel('Likelihood', fontsize=label_size)
    plt.ylabel('# [Normalized]', fontsize=label_size)


    plt.ylim(top=4*np.amax(np.concatenate(n_count)))
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)
