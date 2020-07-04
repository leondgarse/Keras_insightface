try:
    import seaborn as sns

    sns.set(style="darkgrid")
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
import os


def peak_scatter(ax, array, peak_method, color="r", init_epoch=0):
    start = init_epoch + 1
    for ii in array:
        pp = len(ii) - peak_method(ii[::-1]) - 1
        ax.scatter(pp + start, ii[pp], color=color, marker="v")
        ax.text(pp + start, ii[pp], "{:.4f}".format(ii[pp]), va="bottom", ha="right", fontsize=8, rotation=-30)
        start += len(ii)


def arrays_plot(ax, arrays, color=None, label=None, init_epoch=0, pre_value=0):
    tt = []
    for ii in arrays:
        tt += ii
    if pre_value != 0:
        tt = [pre_value] + tt
        xx = list(range(init_epoch, init_epoch + len(tt)))
    else:
        xx = list(range(init_epoch + 1, init_epoch + len(tt) + 1))
    ax.plot(xx, tt, label=label, color=color)
    xticks = list(range(xx[-1]))[:: xx[-1] // 16 + 1]
    # print(xticks, ax.get_xticks())
    if xticks[1] > ax.get_xticks()[1]:
        # print("Update xticks")
        ax.set_xticks(xticks)


def hist_plot(loss_lists, accuracy_lists, customs_dict, loss_names=None, save=None, axes=None, init_epoch=0, pre_item={}, fig_label=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, sharex=True, figsize=(24, 8))
    else:
        fig = axes[0].figure

    if loss_names == None:
        loss_names = [""] * len(loss_lists)

    arrays_plot(axes[0], loss_lists, label=fig_label, init_epoch=init_epoch, pre_value=pre_item.get("loss", 0))
    peak_scatter(axes[0], loss_lists, np.argmin, init_epoch=init_epoch)
    axes[0].set_title("loss")
    if fig_label:
        axes[0].legend(loc="upper right", fontsize=8)

    if len(accuracy_lists) != 0:
        arrays_plot(axes[1], accuracy_lists, label=fig_label, init_epoch=init_epoch, pre_value=pre_item.get("accuracy", 0))
        peak_scatter(axes[1], accuracy_lists, np.argmax, init_epoch=init_epoch)
    axes[1].set_title("accuracy")
    if fig_label:
        axes[1].legend(loc="lower right", fontsize=8)

    # for ss, aa in zip(["lfw", "cfp_fp", "agedb_30"], [lfws, cfp_fps, agedb_30s]):
    for kk, vv in customs_dict.items():
        label =  kk + " - " + fig_label if fig_label else kk
        arrays_plot(axes[2], vv, label=label, init_epoch=init_epoch, pre_value=pre_item.get(kk, 0))
        peak_scatter(axes[2], vv, np.argmax, init_epoch=init_epoch)
    axes[2].set_title(", ".join(customs_dict))
    axes[2].legend(loc="lower right", fontsize=8)

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        mm = (ymax - ymin) * 0.05
        start = init_epoch + 1
        for nn, loss in zip(loss_names, loss_lists):
            ax.plot([start, start], [ymin + mm, ymax - mm], color="k", linestyle="--")
            # ax.text(xx[ss[0]], np.mean(ax.get_ylim()), nn)
            ax.text(start + len(loss) * 0.05, ymin + mm * 4, nn, va="bottom", rotation=-90)
            start += len(loss)

    fig.tight_layout()
    if save != None and len(save) != 0:
        fig.savefig(save)

    last_item = {kk: vv[-1][-1] for kk, vv in customs_dict.items()}
    last_item["loss"] = loss_lists[-1][-1]
    if len(accuracy_lists) != 0:
        last_item["accuracy"] = accuracy_lists[-1][-1]
    return axes, last_item


def hist_plot_split(history, epochs, names=None, customs=[], save=None, axes=None, init_epoch=0, pre_item={}, fig_label=None):
    splits = [[int(sum(epochs[:id])), int(sum(epochs[:id])) + ii] for id, ii in enumerate(epochs)]
    split_func = lambda aa: [aa[ii:jj] for ii, jj in splits if ii < len(aa)]
    if isinstance(history, str):
        history = [history]
    if isinstance(history, list):
        import json

        hh = {}
        for pp in history:
            with open(pp, "r") as ff:
                aa = json.load(ff)
            for kk, vv in aa.items():
                hh.setdefault(kk, []).extend(vv)
        if save != None and len(save) == 0:
            save = os.path.splitext(pp)[0] + ".svg"
    else:
        hh = history.copy()
    loss_lists = split_func(hh.pop("loss"))
    if "accuracy" in hh:
        accuracy_lists = split_func(hh.pop("accuracy"))
    elif "logits_accuracy" in hh:
        accuracy_lists = split_func(hh.pop("logits_accuracy"))
    else:
        accuracy_lists = []
    if len(customs) != 0:
        customs_dict = {kk: split_func(hh[kk]) for kk in customs if kk in hh}
    else:
        hh.pop("lr")
        customs_dict = {kk: split_func(vv) for kk, vv in hh.items()}
    return hist_plot(loss_lists, accuracy_lists, customs_dict, names, save, axes, init_epoch, pre_item, fig_label)
