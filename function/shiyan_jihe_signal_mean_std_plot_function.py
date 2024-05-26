import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
    这个文件的功能是：作为一个函数，提供在标准上制作所有散点图的功能。
    包括，（1）在不同椎间盘结构位置上，制作SI，DH,DHI,HDR标准图谱上的散点图
    （2）在不同的椎间盘退变等级标准谱图上，制作每个结构位置的SI散点图
    What this file does is: As a function, provide the ability to make all
    scatterplots on the standard. Including, (1) Make SI, DH, DHI, HDR scatter
    diagrams on the standard atlas of different intervertebral disc structures
    (2) Make the SI scatter diagram of each structural position on the standard
    spectra of different intervertebral disc degeneration grades

"""
base_dir = "./input/baseline_range"

"""
    保证版本号不能太新，否则报错。安装低版本：pip3 install xlrd==1.2.0
    Make sure that the version number is not too new, otherwise an error will be
    reported. Install the lower version: pip3 install xlrd==1.2.0
"""


def scatter_mean_std(
    sex, age, data, metric="SI", save_path=None
):  # metric=SI, DH, DHI, HDR
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    if metric == "SI":
        # SI grade
        data_SI_excel_name = os.path.join(base_dir, "SI", "SI.xlsx")
        # SI data read
        SI_biaozhun = pd.read_excel(
            data_SI_excel_name, "SI_trend", usecols=[0, 1, 2]
        )

    # data = {'SI': SI, 'DH': DH, 'DHI': DHI, 'HDR': HDR}
    # metrics = ["SI", "DH", "DHI", "HDR"]
    vertebras = ["L1~L2", "L2~L3", "L3~L4", "L4~L5", "L5~S1"]
    # for metric in metrics:
    baseline = dict()
    metric_dir_name = metric if metric != "HDR" else "DWR"
    for vertebra in vertebras:
        baseline[vertebra] = pd.read_excel(
            os.path.join(
                base_dir,
                metric_dir_name,
                str(sex) if metric != "SI" else "",
                f'{metric_dir_name}_{vertebra.replace("~", "")}_trend.xlsx',
            ),
            f"{metric_dir_name} trend",
            usecols=[0, 1] if metric == "SI" else [1, 2, 3],
            header=1 if metric != "SI" else 0,
        )

    # Standard color set, orange for men, blue for women
    # Scatter color settings
    c_point = sns.xkcd_palette(["black"])
    if sex == 1:
        c = sns.xkcd_palette(["orangered"])
    else:
        c = sns.xkcd_palette(["blue"])
    # Scatter shape
    marker = "d"
    # Scatter point abscissa setting, age
    point_age = 5 * (age - 20) / 70
    # The ordinate of the scattered points is the value of each intervertebral
    # disc position
    fig_size = (8, 12)

    # for metric in metrics:
    sns.set_context("talk", font_scale=1, rc={"line.linrwidth": 0.5})
    plt.rc("font", family="monospace")
    fig1 = plt.figure(figsize=fig_size)
    # f_DHI.tight_layout()# Adjust overall whitespace
    fig1.subplots_adjust(wspace=0.3, hspace=0)  # Adjust subplot spacing
    for idx, vertebra in enumerate(vertebras):
        fig1.add_subplot(511 + idx)  # type: ignore
        if idx == 0:
            plt.title(metric)
        ax = sns.lineplot(
            x="age",
            y=vertebra,
            palette=c if metric != "SI" else None,
            hue="gender" if metric != "SI" else None,
            data=baseline[vertebra],
            errorbar="sd",  # type: ignore
        )
        if idx == 0 and metric != "SI":
            plt.legend(loc="center right", bbox_to_anchor=(1, 0.8), ncol=1)
        elif metric != "SI":
            ax.legend_.remove()
        ax.scatter(point_age, data[idx], s=80, marker=marker, c=c_point)
        # if idx != 0 and metric != "SI":
        #     ax.legend_.remove()
    if save_path:
        fig1.savefig(
            os.path.join(
                save_path, f'{metric if metric != "SI" else "SI_weizhi"}.png'
            ),
            bbox_inches="tight",
        )
        # fig.show()

    if metric != "SI":
        return fig1, None

    # Draw scatter points at each level
    sns.set_context("talk", font_scale=1, rc={"line.linrwidth": 0.5})
    plt.rc("font", family="monospace")
    fig2 = plt.figure(figsize=(11.25, 12))
    ax1 = sns.lineplot(
        x="location",
        y="SI",
        hue="grade",
        data=SI_biaozhun,  # type: ignore
        errorbar="sd",  # type: ignore
    )
    for i in range(5):
        ax1.scatter(i, data[i], s=80, marker=marker, c=c_point)
    plt.legend(loc="center right", bbox_to_anchor=(0.98, 0.855), ncol=1)
    if save_path:
        plt.savefig(os.path.join(save_path, "SI_dj.png"), bbox_inches="tight")
        # plt.show()

    return fig1, fig2
