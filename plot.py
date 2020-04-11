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
        pp = peak_method(ii)
        ax.scatter(pp + start, ii[pp], color=color, marker="v")
        ax.text(pp + start, ii[pp], "{:.4f}".format(ii[pp]), va="bottom")
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
    ax.set_xticks(list(range(xx[-1]))[:: xx[-1] // 16 + 1])


def hist_plot(loss_lists, accuracy_lists, customs_dict, loss_names, save="", axes=None, init_epoch=0, pre_item={}):
    if axes is None:
        fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
    else:
        fig = axes[0].figure

    arrays_plot(axes[0], loss_lists, init_epoch=init_epoch, pre_value=pre_item.get("loss", 0))
    peak_scatter(axes[0], loss_lists, np.argmin, init_epoch=init_epoch)
    axes[0].set_title("loss")

    arrays_plot(axes[1], accuracy_lists, init_epoch=init_epoch, pre_value=pre_item.get("accuracy", 0))
    peak_scatter(axes[1], accuracy_lists, np.argmax, init_epoch=init_epoch)
    axes[1].set_title("accuracy")

    # for ss, aa in zip(["lfw", "cfp_fp", "agedb_30"], [lfws, cfp_fps, agedb_30s]):
    for kk, vv in customs_dict.items():
        arrays_plot(axes[2], vv, label=kk, init_epoch=init_epoch, pre_value=pre_item.get(kk, 0))
        peak_scatter(axes[2], vv, np.argmax, init_epoch=init_epoch)
    axes[2].set_title(", ".join(customs_dict))
    axes[2].legend(loc="lower right")

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        mm = (ymax - ymin) * 0.05
        start = init_epoch + 1
        for nn, loss in zip(loss_names, loss_lists):
            ax.plot([start, start], [ymin + mm, ymax - mm], color="k", linestyle="--")
            # ax.text(xx[ss[0]], np.mean(ax.get_ylim()), nn)
            ax.text(start + len(loss) * 0.2, ymin + mm * 4, nn, va='bottom', rotation=-90)
            start += len(loss)

    fig.tight_layout()
    if save != None and len(save) != 0:
        fig.savefig(save)

    last_item = {
        "loss": loss_lists[-1][-1],
        "accuracy": accuracy_lists[-1][-1],
    }
    last_item = {kk: vv[-1][-1] for kk, vv in customs_dict.items()}
    last_item["loss"] = loss_lists[-1][-1]
    last_item["accuracy"] = accuracy_lists[-1][-1]
    return axes, last_item


def hist_plot_split(history, epochs, names, customs=[], save="", axes=None, init_epoch=0, pre_item={}):
    splits = [[int(sum(epochs[:id])), int(sum(epochs[:id])) + ii] for id, ii in enumerate(epochs)]
    split_func = lambda aa: [aa[ii:jj] for ii, jj in splits if ii < len(aa)]
    if isinstance(history, str):
        import json

        with open(history, "r") as ff:
            hh = json.load(ff)
        if save != None and len(save) == 0:
            save = os.path.splitext(history)[0] + ".svg"
    else:
        hh = history.copy()
    loss_lists = split_func(hh.pop("loss"))
    accuracy_lists = split_func(hh.pop("accuracy"))
    if len(customs) != 0:
        customs_dict = {kk: split_func(hh[kk]) for kk in customs if kk in hh}
    else:
        hh.pop("lr")
        customs_dict = {kk: split_func(vv) for kk, vv in hh.items()}
    return hist_plot(loss_lists, accuracy_lists, customs_dict, names, save, axes, init_epoch, pre_item)
