# sort path by time
path_time = [pd.to_datetime("".join(p.stem.split("_")[-4::]), format = "%Y%m%d%H%M") for p in paths]
# [x for _, x in sorted(zip(Y, X))] # sort X by Y values
paths = [p for _, p in sorted(zip(path_time, paths))]
# ========================================================================

if float(lat) >= 0:
    dfc["seasons"] = (dfc.index.month%12 + 3)//3
else:
    dfc["seasons"] = ((dfc.index.month + 6)%12 + 3)//3
# ========================================================================
x = 3.4136
print(f"{x}: .2f")

# ========================================================================
# boxplot

import seaborn as sns

air_poll_names = ['$AOD_{550}$', '$O_3$', '$NO_2$', '$HCHO$']
shap_names = [name + " (SHAP)" for name in air_poll_names]
print(shap_values[:, '$O_3$'].shape, X.shape)
# print(shap_values[:, '$O_3$'].data, X['$O_3$'])
df_analysis = X[air_poll_names].copy()
df_analysis["Season"] = seasons
df_analysis["IGBP"] = igbp_series
df_analysis["ID"] = site_series
for name in air_poll_names:
    df_analysis[name + " (SHAP)"] = shap_values[:, name].values
# print(df_analysis.columns)
dfp = df_analysis.copy()

# site average
dfp = df_analysis.groupby(['ID']).median()
meta = df_analysis[["ID", "IGBP"]].copy()
meta = meta.drop_duplicates(subset='ID', keep="last")
meta = meta.set_index("ID", drop = True)
dfp = pd.concat([dfp, meta], axis = 1)
dfp = dfp.sort_values(by = "IGBP")

# # use pandas to plot manually
# dfp = dfp[[air_poll_names[0], "IGBP"]]
# dfp = pd.pivot_table(dfp, values = air_poll_names[0], index = dfp.index, columns=['IGBP']).reset_index()
# dfp.boxplot()

# use seaborn
dfp = pd.melt(dfp, id_vars=['IGBP'], value_vars=shap_names,
        var_name='Pollutant', value_name="$SHAP \; (gC \; m^{-2} \; d^{-1})$"
        )

# sns.set_style("white")
sns.set(rc={'figure.figsize':(12, 10)}, style = "white", font_scale = 1.5)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
ax = sns.boxplot(x="IGBP", y="$SHAP \; (gC \; m^{-2} \; d^{-1})$", hue="Pollutant",
                 data=dfp, palette="Set3")
ax.axhline(y=0.0, color='k', linestyle='--', alpha = 0.5)
ax.set_ylim(-0.3, 0.3)
plt.tick_params(direction = "in", which = "both")
fig = ax.get_figure()

# Python > 3.7
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc = "upper center", framealpha = 0.1, frameon = True , bbox_to_anchor=(0.5, 1.1), ncol = 4)

# ============================================================================
# split one column into three and rename them
df = df.join(df["date"].str.split("_", expand=True).astype(int).rename(columns = {0: "year", 1: "month", 2: "day"}))

# ============================================================================
# vertical barplots

# site wise r2

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

df = read_df_dn()
df["site"] = df["site"] + "(" + df["igbp"] + ")"

time_type = "diel"

# print(df.columns)

df["total"] = df[f"MDS_{time_type}_r2"] + df[f"RFR_{time_type}_r2"]
df["dif_r2"] = df[f"RFR_{time_type}_r2"] - df[f"MDS_{time_type}_r2"]
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(8, 18))

df = df.sort_values("total", ascending=False)
# split fig
df = df.iloc[100::, :]
print(len(df))

# Plot the total crashes
sns.set_color_codes("pastel")
g = sns.barplot(x="total", y="site", data=df,
            label=f"MDS $r^2$", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x=f"RFR_{time_type}_r2", y="site", data=df,
            label=f"RFR $r^2$", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 2), ylabel="",
       xlabel=f"$The \; r^2 \; comparison$")
sns.despine(left=True, bottom=True)

for i in range(len(df)):
    p = ax.patches[i]
    height = p.get_height()
    width = p.get_width()
    if p.get_x() + width >= 8:
        x_pos = 8
    else:
        x_pos = p.get_x() + width
    # ax.text(x_pos, p.get_y() + height/2, np.round(df["dif_bias"].values[i], 2))
    ax.annotate(f"{np.round(df['dif_r2'].values[i], 2)}", (x_pos, p.get_y() + height))
    

f.savefig(f"3drivers/site_r2_compare_sort_dif_{time_type}_b.pdf", dpi = 600, format = "pdf", bbox_inches = "tight")

# ========================================================================================================================
# drop rows with duplicated index
df3 = df3[~df3.index.duplicated(keep='first')]

# ========================================================================================================================