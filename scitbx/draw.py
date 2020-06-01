from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, linregress

def main():
    plt.rcParams['font.sans-serif'] = ['serif']
    sns.set(context="paper", style="ticks",
            font="serif", font_scale=2, rc={"figure.figsize": (12, 8)})

    # # normalize
    # p_flag = Path("norm_daily_avg.csv")
    # if not p_flag.exists():
    #     p = Path("daily_avg.csv")
    #     df = pd.read_csv(p, index_col=0)
    #     df = df.fillna(method = "backfill")

    #     columns = df.columns.tolist()
    #     for col_name in columns:
    #         timeseries = df[col_name].values
    #         scaler = MinMaxScaler(feature_range=(0, 1))
    #         df[col_name] = scaler.fit_transform(timeseries.reshape(-1, 1))
        
    #     # save normalized daily averaged data to csv file
    #     df.to_csv("norm_daily_avg.csv")
    # else:
    #     df = pd.read_csv(p_flag, index_col=0)
    #     columns = df.columns.tolist()

    p = Path("daily_avg.csv")
    df = pd.read_csv(p, index_col=0)
    df = df.fillna(method = "backfill")

    columns = df.columns.tolist()

    # for each site code: {
    codes = [col.split("_")[0] for col in columns]
    # remove duplicates
    new_codes = []
    for code in codes:
        if not code in new_codes:
            new_codes.append(code)
    codes = new_codes
    del(new_codes)

    for code in codes:
        site_name = code + "_site"
        naq_name = code + "_naq"

        site = df[site_name]
        naq = df[naq_name] * 1.96 # ppb to ug/m3
        # print(site, naq)
        p_score = pearsonr(site, naq)
        print(f"pearson: {np.round(p_score[0], 4)}, p_value: {np.round(p_score[1], 4)}")
        slope, intercept, r_value, p_value, stderr = linregress(site, naq)
        print(f"r2 score is {r_value**2}; p_value is {p_value}")

        fig, ax = plt.subplots()
        ax.plot(df.index, site, "g*-", lw = 2, label = "Site (daily)")
        ax.plot(df.index, naq, "bo-", lw = 2, label = "NAQ-PMS (daily)")
        ax.set_xticks(ax.get_xticks()[::40])
        ax.tick_params(axis='x', rotation=20)
        ax.legend()
        ax.set_title(fr"{code} ($ \mu g/m^{3}$)", fontsize = 24, color='k')
        # plt.show()
        plt.savefig(code + ".pdf", dpi = 600, format = "pdf")
        # exit(0)

    site = df.select(lambda col: col.endswith('_site'), axis = 1)
    site = site.mean(axis = 1)

    naq = df.select(lambda col: col.endswith('_naq'), axis = 1)
    naq = naq.mean(axis = 1) * 1.96 # ppb to ug/m3

    # df_whole_city = pd.concat([df_site, df_naq], axis = 1)
    # df_whole_city.columns = ["site", "naq"]
    p_score = pearsonr(site, naq)
    print(f"pearson: {np.round(p_score[0], 4)}, p_value: {np.round(p_score[1], 4)}")
    slope, intercept, r_value, p_value, stderr = linregress(site, naq)
    print(f"for whole city, r2 score is {r_value**2}; p_value is {p_value}")

    fig, ax = plt.subplots()
    ax.plot(df.index, site, "g*-", lw = 2, label = "Site based (daily)")
    ax.plot(df.index, naq, "bo-", lw = 2, label = "NAQ-PMS (daily)")
    ax.set_xticks(ax.get_xticks()[::40])
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    ax.set_title(fr"Chongqing Ozone timeseries ($ \mu g/m^{3}$)", fontsize = 24, color='k')
    # plt.show()
    plt.savefig("1499A.pdf", dpi = 600, format = "pdf")

    # }
    

if __name__ == "__main__":
    main()