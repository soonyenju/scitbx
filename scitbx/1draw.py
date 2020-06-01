from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    plt.rcParams['font.sans-serif'] = ['serif']
    sns.set(context="paper", style="ticks",
            font="serif", font_scale=2, rc={"figure.figsize": (12, 8)})
    paths = Path(".").glob(r"*xsites*")
    for p in paths:
        print(p.stem)
        out_path = p.joinpath("figs")
        if not out_path.exists(): out_path.mkdir()
        res_paths = p.joinpath("1data4pics").glob(r"*A*")
        for rp in res_paths:
            print(rp.stem)
            df = pd.read_csv(rp)
            draw_coorelation(out_path, rp.stem, df, "pred", "ys")

        # exit(0)

def draw_coorelation(out_path, name, df, key1, key2):
    fig = plt.figure() 

    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[key1], df[key2])
    # use line_kws to set line label for legend
    ax = sns.regplot(x = key1, y = key2, data = df, color = 'k',
        line_kws={'label':f"Y = {np.round(slope, 2)} * X + {np.round(intercept, 2)}"})
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    # plot legend
    # ax.legend()

    plt.text(5, 240, f"y = {np.round(slope, 2)} * x + {np.round(intercept, 2)}", size = 20,\
            family = "serif", color = "k", style = "italic", weight = "light",\
         bbox = dict(facecolor = "w", alpha = 0.2))

    plt.text(5, 220, f"r2: {np.round(r_value ** 2, 2)}", size = 20,\
            family = "serif", color = "k", style = "italic", weight = "light",\
         bbox = dict(facecolor = "w", alpha = 0.2))

    plt.text(5, 200, f"p: {np.round(p_value, 2)}", size = 20,\
            family = "serif", color = "k", style = "italic", weight = "light",\
         bbox = dict(facecolor = "w", alpha = 0.2))

    plt.text(5, 180, f"std: {np.round(std_err, 2)}", size = 20,\
            family = "serif", color = "k", style = "italic", weight = "light",\
         bbox = dict(facecolor = "w", alpha = 0.2))
    ax.set_xlabel(r"Predicted $O_3$")
    ax.set_ylabel(r"Observed $O_3$")

    fig.set_figheight(8)
    fig.set_figwidth(12)
    fig.set_size_inches((12, 8))
    fig.tight_layout()
    fig.subplots_adjust(wspace=.02, hspace=.02, right = 0.95, top = 0.92)
    if name == "1499A":
        title_name = "Chongqing"
    else:
        title_name = name
    ax.set_title(title_name, fontsize = 24, color='k')
    # plt.show()
    # exit(0)
    plt.savefig(out_path.joinpath(name + ".pdf"), dpi = 600, format = "pdf")

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

if __name__ == "__main__":
    main()