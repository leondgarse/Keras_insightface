import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from cycler import cycler

plt.style.use("seaborn")
EVALS_NAME = ["lfw", "cfp_fp", "agedb_30", "IJBB", "IJBC"]
EVALS_LINE_STYLE = ["-", "--", "-.", ":"]
MAX_COLORS = 10
Scale = 1
Default_legend_font_size = 8
Default_text_font_size = 9
Default_figure_base_size = 8
# COLORS = cm.rainbow(np.linspace(0, 1, MAX_COLORS))

SPLIT_LINES = {}

try:
    import seaborn as sns

    sns.set(style="darkgrid")
    COLORS = sns.color_palette("deep", n_colors=MAX_COLORS)
except:
    pass


def set_colors(max_color, palette="deep"):
    print("Available palette names: deep, muted, bright, pastel, dark, colorblind, rainbow")
    global MAX_COLORS
    global COLORS
    MAX_COLORS = max_color
    if palette == "rainbow":
        COLORS = cm.rainbow(np.linspace(0, 1, MAX_COLORS))
    else:
        COLORS = sns.color_palette(palette, n_colors=MAX_COLORS)


def set_scale(scale):
    global Scale, Default_text_font_size, Default_legend_font_size, Default_figure_base_size
    if Scale != scale:
        import matplotlib as mpl

        Scale, scale = scale, scale / Scale
        mpl.rcParams["axes.titlesize"] *= scale
        mpl.rcParams["axes.labelsize"] *= scale
        mpl.rcParams["axes.labelpad"] *= scale
        mpl.rcParams["legend.fontsize"] *= scale
        mpl.rcParams["font.size"] *= scale

        mpl.rcParams["axes.linewidth"] *= scale
        mpl.rcParams["lines.linewidth"] *= scale
        mpl.rcParams["grid.linewidth"] *= scale
        mpl.rcParams["lines.markersize"] *= scale

        mpl.rcParams["xtick.labelsize"] *= scale
        mpl.rcParams["ytick.labelsize"] *= scale
        mpl.rcParams["ytick.major.pad"] *= scale
        mpl.rcParams["xtick.major.pad"] *= scale

        Default_text_font_size *= scale
        Default_legend_font_size *= scale
        Default_figure_base_size *= scale


def peak_scatter(ax, array, peak_method, color="r", init_epoch=0, limit_max=1e9, limit_min=0):
    start = init_epoch + 1
    for ii in array:
        pp = len(ii) - peak_method(ii[::-1]) - 1
        # Skip scatter if it's 0
        if ii[pp] != 0:
            y_pos = ii[pp - 1] if np.isnan(ii[pp]) and pp != 0 else ii[pp]
            y_pos = y_pos if y_pos < limit_max else limit_max
            y_pos = y_pos if y_pos > limit_min else limit_min
            ax.scatter(pp + start, y_pos, color=color, marker="v")
            ax.text(
                pp + start,
                y_pos,
                "Nan" if np.isnan(ii[pp]) else "{:.4f}".format(ii[pp]),
                va="bottom",
                ha="right",
                fontsize=Default_text_font_size,
                rotation=-30,
            )
        start += len(ii)


def arrays_plot(ax, arrays, color=None, label=None, init_epoch=0, pre_value=0, linestyle="-", limit_max=1e9, limit_min=0):
    tt = []
    for ii in arrays:
        tt += ii
    if pre_value != 0 and init_epoch == 0:
        tt[0] = pre_value
        xx = list(range(init_epoch + 1, init_epoch + len(tt) + 1))
    if pre_value != 0 and init_epoch != 0:
        tt = [pre_value] + tt
        xx = list(range(init_epoch, init_epoch + len(tt)))
    else:
        xx = list(range(init_epoch + 1, init_epoch + len(tt) + 1))
    # Replace 0 values with their previous element
    for id, ii in enumerate(tt):
        if (ii == 0 or np.isnan(ii)) and id != 0:
            tt[id] = tt[id - 1]
        if ii > limit_max:
            tt[id] = limit_max
        if ii < limit_min:
            tt[id] = limit_min
    ax.plot(xx, tt, label=label, color=color, linestyle=linestyle)
    xticks = list(range(xx[-1]))[:: xx[-1] // 16 + 1]
    # print(xticks, ax.get_xticks())
    if len(xticks) > 1 and xticks[1] > ax.get_xticks()[1]:
        # print("Update xticks")
        ax.set_xticks(xticks)


def hist_plot(
    loss_lists,
    accuracy_lists,
    customs_dict,
    loss_names=None,
    save=None,
    axes=None,
    init_epoch=0,
    pre_item={},
    fig_label=None,
    eval_split=True,
    limit_loss_max=1e9,
):
    global SPLIT_LINES
    if axes is None:
        if eval_split:
            fig, axes = plt.subplots(2, 3, sharex=False, figsize=(3 * Default_figure_base_size, 2 * Default_figure_base_size))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, sharex=False, figsize=(3 * Default_figure_base_size, 1 * Default_figure_base_size))
        for ax in axes:
            ax.set_prop_cycle(cycler("color", COLORS))
        SPLIT_LINES = {}
    else:
        fig = axes[0].figure
        # Empty titles
        for ax in axes:
            ax.set_title("")
        eval_split = True if len(axes) == 6 else False
    axes = axes.tolist()

    if len(loss_lists) != 0:
        arrays_plot(
            axes[0],
            loss_lists,
            label=fig_label,
            init_epoch=init_epoch,
            pre_value=pre_item.get("loss", 0),
            limit_max=limit_loss_max,
        )
        peak_scatter(axes[0], loss_lists, np.argmin, init_epoch=init_epoch, limit_max=limit_loss_max)
        # if axes[0].get_ylim()[0] < 0:
        #     axes[0].set_ylim(0)
    axes[0].set_title("loss")
    if fig_label:
        axes[0].legend(loc="upper right", fontsize=Default_legend_font_size)

    cur_color = axes[0].lines[-1].get_color()
    if len(accuracy_lists) != 0:
        arrays_plot(
            axes[1],
            accuracy_lists,
            color=cur_color,
            label=fig_label,
            init_epoch=init_epoch,
            pre_value=pre_item.get("accuracy", 0),
        )
        peak_scatter(axes[1], accuracy_lists, np.argmax, init_epoch=init_epoch)
    axes[1].set_title("accuracy")
    if fig_label:
        axes[1].legend(loc="lower right", fontsize=Default_legend_font_size)

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
            eval_id += 0 if eval_split and eval_ax != 5 else 1
            eval_ax = min(eval_ax + eval_ax_step, 5)
        else:
            ax = axes[other_custom_ax]
            title = kk if len(ax.get_title()) == 0 else ax.get_title() + ", " + kk
            ax.set_title(title)
            linestyle = EVALS_LINE_STYLE[other_custom_id + 0 if eval_split else 1]
            other_custom_id += 1
        label = kk + " - " + fig_label if fig_label else kk
        arrays_plot(ax, vv, color=cur_color, label=label, init_epoch=init_epoch, pre_value=pre_item.get(kk, 0), linestyle=linestyle)
        peak_scatter(ax, vv, np.argmax, init_epoch=init_epoch)

    # eval_ax = eval_ax if eval_split else eval_ax + 1
    for ii in range(eval_ax_start, eval_ax + 1):
        axes[ii].legend(loc="lower left", fontsize=Default_legend_font_size)

    if len(axes) > 3 and other_custom_id > 0:
        axes[other_custom_ax].legend(loc="lower right", fontsize=Default_legend_font_size)

    # cur_color = "k" if len(axes[0].lines) == 1 else axes[0].lines[-1].get_color()

    for ax_id, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        mm = (ymax - ymin) * 0.05
        start = init_epoch + 1
        for loss_id, loss in enumerate(loss_lists):
            # ax.plot([start, start], [ymin + mm, ymax - mm], color="k", linestyle="--")
            split_lines = ax.plot([start, start], [ymin + mm, ymax - mm], color=cur_color, linestyle="--")

            if loss_names is not None and len(loss_names) > loss_id:
                nn = loss_names[loss_id]
                # ax.text(start + len(loss) * 0.05, np.mean(ax.get_ylim()), nn, va="top", rotation=-90, fontweight="roman", c=cur_color)
                # ax.text(start + len(loss) * 0.05, ymax - mm * 4, nn, va="top", rotation=-90, fontweight="roman", c=cur_color)
                ax.text(start + len(loss) * 0.05, ymin + mm * 4, nn, va="bottom", rotation=-90, fontweight="roman", c=cur_color)

            split_line_id = "{}_{}".format(ax_id, start)
            if split_line_id in SPLIT_LINES:
                SPLIT_LINES[split_line_id].remove()
            SPLIT_LINES[split_line_id] = split_lines[0]
            start += len(loss)

    fig.tight_layout()
    if save != None and len(save) != 0:
        fig.savefig(save)

    last_item = {kk: vv[-1][-1] for kk, vv in customs_dict.items()}
    if len(loss_lists) != 0:
        last_item["loss"] = loss_lists[-1][-1]
    if len(accuracy_lists) != 0:
        last_item["accuracy"] = accuracy_lists[-1][-1]
    return np.array(axes), last_item


def hist_plot_split(
    history,
    epochs=[100],
    names=None,
    customs=EVALS_NAME[:3] + ["lr"],
    save=None,
    axes=None,
    init_epoch=0,
    pre_item={},
    fig_label=None,
    eval_split=True,
    limit_loss_max=1e9,
    skip_epochs=0,
):
    splits = [[int(sum(epochs[:id])), int(sum(epochs[:id])) + ii] for id, ii in enumerate(epochs)]
    if skip_epochs != 0:
        splits = [[ss, ee] for ss, ee in splits if ee > skip_epochs - init_epoch]
        splits[0][0] = max(splits[0][0], skip_epochs - init_epoch)
        if names is not None and len(names) != 0:
            names = names[-len(splits) :]
        if init_epoch < skip_epochs:
            init_epoch = skip_epochs

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

    if fig_label is None and isinstance(history[-1], str):
        fig_label = os.path.splitext(os.path.basename(history[-1]))[0]

    return hist_plot(
        loss_lists,
        accuracy_lists,
        customs_dict,
        names,
        save,
        axes,
        init_epoch,
        pre_item,
        fig_label,
        eval_split=eval_split,
        limit_loss_max=limit_loss_max,
    )


def choose_accuracy(aa, skip_name_len=0, sort_metric=False, metric_key="agedb_30", key_picks=EVALS_NAME):
    import json
    import pandas as pd

    # key_picks = ['lfw', 'cfp_fp', 'agedb_30']
    dd_metric_max, dd_all_max, dd_sum_max, dd_latest = {}, {}, {}, {}
    metric_key = metric_key if metric_key in key_picks else key_picks[-1]  # Pick -1 item if metric_key not in key_picks
    for pp in aa:
        with open(pp, "r") as ff:
            hh = json.load(ff)
        nn = os.path.splitext(os.path.basename(pp))[0][skip_name_len:]
        metric_arg_max = np.argmax(hh[metric_key])
        dd_metric_max[nn] = {kk: hh[kk][metric_arg_max] for kk in key_picks if kk in hh}
        dd_metric_max[nn]["epoch"] = int(metric_arg_max)

        dd_all_max[nn] = {kk: "%.4f, %2d" % (max(hh[kk]), np.argmax(hh[kk])) for kk in key_picks if kk in hh}
        # dd_all_max[nn] = {kk: max(hh[kk]) for kk in key_picks}
        # dd_all_max[nn].update({kk + "_epoch": np.argmax(hh[kk]) for kk in key_picks})

        sum_arg_max = np.argmax(np.sum([hh[kk] for kk in key_picks if kk in hh], axis=0))
        dd_sum_max[nn] = {kk: hh[kk][sum_arg_max] for kk in key_picks if kk in hh}
        dd_sum_max[nn]["epoch"] = int(sum_arg_max)
        dd_latest[nn] = {kk: hh[kk][-1] for kk in key_picks if kk in hh}
        dd_latest[nn]["epoch"] = len(hh[metric_key])

    names = [metric_key + " max", "all max", "sum max", "latest"]
    for nn, dd in zip(names, [dd_metric_max, dd_all_max, dd_sum_max, dd_latest]):
        print()
        print(">>>>", nn, ":")
        # print(pd.DataFrame(dd).T.to_markdown())
        rr = pd.DataFrame(dd).T
        rr = rr.sort_values(metric_key) if sort_metric else rr
        print(rr.to_markdown())
    return dd_metric_max, dd_all_max, dd_sum_max
