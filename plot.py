import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from cycler import cycler

plt.style.use("seaborn")
EVALS_NAME = ["lfw", "cfp_fp", "agedb_30"]
EVALS_LINE_STYLE = ["-", "--", "-.", ":"]
MAX_COLORS = 10
# COLORS = cm.rainbow(np.linspace(0, 1, MAX_COLORS))

try:
    import seaborn as sns
    sns.set(style="darkgrid")
    COLORS = sns.color_palette('deep', n_colors=MAX_COLORS)
except:
    pass

def set_colors(max_color, palette='deep'):
    print("Available palette names: deep, muted, bright, pastel, dark, colorblind, rainbow")
    global MAX_COLORS
    global COLORS
    MAX_COLORS = max_color
    if palette == 'rainbow':
        COLORS = cm.rainbow(np.linspace(0, 1, MAX_COLORS))
    else:
        COLORS = sns.color_palette(palette, n_colors=MAX_COLORS)

def peak_scatter(ax, array, peak_method, color="r", init_epoch=0):
    start = init_epoch + 1
    for ii in array:
        pp = len(ii) - peak_method(ii[::-1]) - 1
        # Skip scatter if it's 0
        if ii[pp] != 0:
            ax.scatter(pp + start, ii[pp], color=color, marker="v")
            ax.text(pp + start, ii[pp], "{:.4f}".format(ii[pp]), va="bottom", ha="right", fontsize=8, rotation=-30)
        start += len(ii)


def arrays_plot(ax, arrays, color=None, label=None, init_epoch=0, pre_value=0, linestyle="-"):
    tt = []
    for ii in arrays:
        tt += ii
    if pre_value != 0:
        tt = [pre_value] + tt
        xx = list(range(init_epoch, init_epoch + len(tt)))
    else:
        xx = list(range(init_epoch + 1, init_epoch + len(tt) + 1))
    # Replace 0 values with their previous element
    for id, ii in enumerate(tt):
        if ii == 0 and id != 0:
            tt[id] = tt[id - 1]
    ax.plot(xx, tt, label=label, color=color, linestyle=linestyle)
    xticks = list(range(xx[-1]))[:: xx[-1] // 16 + 1]
    # print(xticks, ax.get_xticks())
    if len(xticks) > 1 and xticks[1] > ax.get_xticks()[1]:
        # print("Update xticks")
        ax.set_xticks(xticks)


def hist_plot(
    loss_lists, accuracy_lists, customs_dict, loss_names=None, save=None, axes=None, init_epoch=0, pre_item={}, fig_label=None, eval_split=False
):
    if axes is None:
        if eval_split:
            fig, axes = plt.subplots(2, 3, sharex=True, figsize=(24, 16))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, sharex=True, figsize=(24, 8))
        for ax in axes:
            ax.set_prop_cycle(cycler('color', COLORS))
    else:
        fig = axes[0].figure
        # Empty titles
        for ax in axes:
            ax.set_title("")
        if len(axes) == 6:
            eval_split = True
    axes = axes.tolist()
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

    other_customs = [ii for ii in customs_dict if ii not in EVALS_NAME]
    if len(other_customs) != 0:
        if len(axes) == 3:
            axes.append(axes[0].twinx())
        if len(axes) == 4:
            axes[-1].set_title(axes[0].get_title())
            axes[0].set_title("")

    if eval_split:
        # Plot two rows, 2 x 3
        other_custom_ax = 2
        eval_ax_start, eval_ax = 3, 3
        eval_ax_step = 1
    else:
        # Plot one row, 1 x 3
        other_custom_ax = 3
        eval_ax_start, eval_ax = 2, 2
        eval_ax_step = 0

    # Keep the same color, but different from line styles.
    eval_id, other_custom_id = 0, 0
    for kk, vv in customs_dict.items():
        if kk in EVALS_NAME:
            ax = axes[eval_ax]
            title = kk if len(ax.get_title()) == 0 else ax.get_title() + ", " + kk
            ax.set_title(title)
            linestyle = EVALS_LINE_STYLE[eval_id]
            cur_color = ax.lines[-1].get_color() if eval_id != 0 else None
            eval_id += 0 if eval_split else 1
            eval_ax += eval_ax_step
        else:
            ax = axes[other_custom_ax]
            title = kk if len(ax.get_title()) == 0 else ax.get_title() + ", " + kk
            ax.set_title(title)
            linestyle = EVALS_LINE_STYLE[other_custom_id + 0 if eval_split else 1]
            cur_color = ax.lines[-1].get_color() if other_custom_id != 0 else None
            other_custom_id += 1
        label = kk + " - " + fig_label if fig_label else kk
        arrays_plot(ax, vv, color=cur_color, label=label, init_epoch=init_epoch, pre_value=pre_item.get(kk, 0), linestyle=linestyle)
        peak_scatter(ax, vv, np.argmax, init_epoch=init_epoch)

    eval_ax = eval_ax if eval_split else eval_ax + 1
    for ii in range(eval_ax_start, eval_ax):
        axes[ii].legend(loc="lower left", fontsize=8)

    if len(axes) > 3 and other_custom_id > 0:
        axes[other_custom_ax].legend(loc="lower right", fontsize=8)

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
    return np.array(axes), last_item


def hist_plot_split(history, epochs, names=None, customs=[], save=None, axes=None, init_epoch=0, pre_item={}, fig_label=None, eval_split=True):
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
        # hh.pop("lr")
        customs_dict = {kk: split_func(vv) for kk, vv in hh.items() if kk in EVALS_NAME}
    return hist_plot(loss_lists, accuracy_lists, customs_dict, names, save, axes, init_epoch, pre_item, fig_label, eval_split=eval_split)
