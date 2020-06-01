from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Painter():
    def __init__(self, figsize = (12, 8)):
        plt.rcParams['font.sans-serif'] = ['serif']
        sns.set(context="paper", style="ticks",
            font="serif", font_scale=2, rc={"figure.figsize": figsize}
        )

    def scatter_regression(self, save_folder, name, df, key1, key2, x_lim = [-35, 60], y_lim = [-35, 60], text1pos = [-25, 50], text2pos = [-25, 40]):
        fig = plt.figure() 

        # get coeffs of linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[key1], df[key2])
        # use line_kws to set line label for legend
        ax = sns.regplot(x = key1, y = key2, data = df, color = 'k',
            line_kws={'label':f"Y = {np.round(slope, 2)} * X + {np.round(intercept, 2)}"})
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        # plot legend
        # ax.legend()

        plt.text(*text1pos, f"y = {np.round(slope, 2)} * x + {np.round(intercept, 2)}", size = 20,\
                family = "serif", color = "k", style = "italic", weight = "light",\
             bbox = dict(facecolor = "w", alpha = 0.2))

        plt.text(*text2pos, f"r2: {np.round(r_value ** 2, 2)}", size = 20,\
                family = "serif", color = "k", style = "italic", weight = "light",\
             bbox = dict(facecolor = "w", alpha = 0.2))

        # plt.text(-25, 30, f"p: {np.round(p_value, 2)}", size = 20,\
        #         family = "serif", color = "k", style = "italic", weight = "light",\
        #      bbox = dict(facecolor = "w", alpha = 0.2))

        # plt.text(-25, 20, f"std: {np.round(std_err, 2)}", size = 20,\
        #        family = "serif", color = "k", style = "italic", weight = "light",\
        #     bbox = dict(facecolor = "w", alpha = 0.2))
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)

        fig.set_figheight(8)
        fig.set_figwidth(12)
        fig.set_size_inches((12, 8))
        fig.tight_layout()
        fig.subplots_adjust(wspace=.02, hspace=.02, right = 0.95, top = 0.92)

        title_name = name
        ax.set_title(title_name, fontsize = 24, color='k')
        plt.savefig(save_folder.joinpath(name + ".pdf"), dpi = 600, format = "pdf")

    def lineplot(self, save_folder, name, df, key1, key2):
        fig, ax = plt.subplots()
        ax.plot(df.index, df[key1], "g*-", lw = 2, label = key1)
        ax.plot(df.index, df[key2], "bo-", lw = 2, label = key2)
        # ax.set_xticks(ax.get_xticks()[::40])
        ax.tick_params(axis='x', rotation=20)
        ax.legend()
        title_name = name
        ax.set_title(name, fontsize = 24, color='k')
        plt.savefig(save_folder.joinpath(name + ".pdf"), dpi = 600, format = "pdf")
        