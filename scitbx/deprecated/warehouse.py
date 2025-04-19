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
df = df[~df.index.duplicated(keep='first')]
df = df.loc[:,~df.columns.duplicated()]

# ========================================================================================================================
# x-axis datetime format
import matplotlib.dates as mdates
dtfmt = mdates.DateFormatter('%d/%m/%y %H:%m')
for ax in axes:
    ax.xaxis.set_major_formatter(dtfmt)
# =========================================================================================================================
# Diurnal plots

for df_orig, site_name in zip([seb, sab], ["Sebungan", "Sabaju"]):
    df_orig = df_orig.replace(-9999., np.nan)
    for method in df_orig.columns:

        df = df_orig[[method]].copy()
        df['Time'] = df.index.map(lambda x: x.strftime("%H:%M"))
        df['Month'] = df.index.map(lambda x: x.strftime("%m"))
        df['Year'] = df.index.map(lambda x: x.strftime("%Y"))
        
        plot_diurnal(df, method, site_name)
        
print("done")

import matplotlib.pyplot as plt

def plot_diurnal(df, method, site_name):
    df_diurnal = df.pivot_table(method, ['Year', 'Month', 'Time'], aggfunc='mean').reset_index()
    months = list(df_diurnal["Month"].unique())
    months.sort(key=float)

    fig, axs = plt.subplots(figsize=(16, 12), 
                            nrows = 4, ncols = 3,
                            sharex = True,
                            sharey = True
                           )

    for mt, ax in zip(months, axs.flatten()):
        ax.tick_params(direction = "in")
        mt_label = datetime.strptime(mt, "%m").strftime("%B")
        dft = df_diurnal.query("Month == @mt")
        # years = dft["Year"].unique()
        dft = dft.drop("Month", axis = 1)
        dft = dft.set_index("Time")
        # print(dft)
        # # ==================================================================
        # # method 1: plot lines in one subfigure automatedly but has issues with legend
        # dft.groupby("Year")[method].plot(legend=True, ax=ax, title=mt_label)
        
        # # ==================================================================
        # # method 2: plot lines and legend manually
        years = [2016, 2017, 2018, 2019, 2020]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        legends = [str(lg) for lg in years]
        for yr, c in zip(years, colors):
            if dft[dft["Year"] == str(yr)].empty: continue
            dft[dft["Year"] == str(yr)].plot(color = c, ax = ax, legend = None, title=mt_label)
            # lgds.append(str(yr))
        # ax.legend(lgds)
        ax.set_ylabel("NEE $(\mu mol \; m^{-2} \; d^{-1})$")
        ax.set_xlabel("Hour", fontsize = 12)
    #     break
    
    # =========================================================================================================
    # extract common legends
    lines = []
    labels = []

    ax_colors = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        for line2D in axLine:
            if line2D.get_color() in ax_colors:
                continue
            else:
                ax_colors.append(line2D.get_color())
                line2D.set_label(
                    dict(zip(colors, legends))[line2D.get_color()]
                )
                lines.append(line2D)
    lines = sorted(lines, key=lambda x: int(x.get_label()), reverse=False)
    axs.flatten()[-2].legend(handles = lines, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(lines))
    # =========================================================================================================
    plt.suptitle(method.upper(), fontsize = 16, x = 0.15, y = 0.93)

    plt.savefig(f"{site_name}-{method}-diurnal.png", bbox_inches = "tight", dpi = 300)
    
    plt.close(fig)
# =========================================================================================================================
# increase marker size in legend
legend = ax.legend(loc = "lower left")
for handle in legend.legendHandles:
    handle.set_sizes([20])
# =========================================================================================================================
# tick rotation
ax.tick_params(axis='x', rotation=45)
# =========================================================================================================================
# Efficient way to group indices of the same elements in a list
data = [1, 2, 2, 5, 8, 3, 3, 9, 0, 1]
pd.Series(range(len(data))).groupby(data, sort=False).apply(list).tolist()
# [[0, 9], [1, 2], [3], [4], [5, 6], [7], [8]]

# =========================================================================================================================
# Intersection of multiple lists
from functools import reduce
# d = [list1, list2, ...]
# solution1:
set.intersection(*map(set,d))
# solution2:
# apply intersect1d to (a list of) multiple lists:
reduce(np.intersect1d, d)


# =========================================================================================================================
'''
Cirular bar plot
'''
# https://www.python-graph-gallery.com/circular-barplot-with-groups

OFFSET = np.pi / 2

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 


# Ensures reproducibility of random numbers
rng = np.random.default_rng(123)

# Build a dataset
df = pd.DataFrame({
    "name": [f"item {i}" for i in range(1, 51)],
    "value": rng.integers(low=30, high=100, size=50),
    "group": ["A"] * 10 + ["B"] * 20 + ["C"] * 12 + ["D"] * 8
})

# # Build a dataset
# df = pd.DataFrame({
#     "name": [f"item {i}" for i in range(1, 32 * 6 + 1)],
#     "value": rng.integers(low=30, high=100, size=32 * 6),
#     "group": ["A"] * 32 + ["B"] * 64 + ["C"] * 32 + ["D"] * 64
# })

# Show 3 first rows
df.head(3)


# All this part is like the code above
VALUES = df["value"].values
LABELS = df["name"].values
GROUP = df["group"].values

PAD = 3
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

offset = 0
IDXS = []
GROUPS_SIZE = [10, 20, 12, 8]
# GROUPS_SIZE = [32, 64, 32, 64]
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
ax.set_theta_offset(OFFSET)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# GROUPS_SIZE = [10, 20, 12, 8]
COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
    edgecolor="white", linewidth=2
)

add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Extra customization below here --------------------

# This iterates over the sizes of the groups adding reference
# lines and annotations.

offset = 0 
for group, size in zip(["A", "B", "C", "D"], GROUPS_SIZE):
    # Add line below bars
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
    ax.plot(x1, [-5] * 50, color="#333333")
    
    # Add text to indicate group
    ax.text(
        np.mean(x1), -20, group, color="#333333", fontsize=14, 
        fontweight="bold", ha="center", va="center"
    )
    
    # Add reference lines at 20, 40, 60, and 80
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
    ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)
    
    offset += size + PAD


# =========================================================================================================================
# Transform data to axis coordinates:

ax.transData.transform((2, 15))


# =========================================================================================================================
# element-wise add two arrays ignoring NaNs
# https://stackoverflow.com/questions/33269369/adding-two-2d-numpy-arrays-ignoring-nans-in-them
np.nansum(np.dstack((A,B)),2)


# =========================================================================================================================
# Offset time
df02.index = df02.index + pd.tseries.offsets.Minute(1)
pd.DateOffset(months=1)

# ==========================================================================================================================
# list comprehension if else
[x+1 if x >= 45 else x+5 for x in l]

# ========================================================================================================================================================
# matplotlib default colors
['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# ===============================================================================================================================
# overlapping subplots tick labels
# https://stackoverflow.com/questions/15773049/remove-overlapping-tick-marks-on-subplot-in-matplotlib
from matplotlib.ticker import MaxNLocator # added
nbins = len(ax1.get_xticklabels()) # added 
ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper')) # added 
# or
ax.xaxis.set_major_locator(plt.MaxNLocator(3))


axes[2].yaxis.set_label_coords(-0.12, -0.05) # , transform = axes[2].transAxes
axes[1].yaxis.set_label_coords(-0.05, -0.05, transform = axes[1].transAxes)

axes[7].set_axis_off()

axes[5].xaxis.set_tick_params(labelbottom=True)

# ===================================================================================================================
ax.tick_params(axis='x', rotation=45)

# ===================================================================================================================================================
lgnd = ax.legend(loc = "lower left", framealpha = 0.1, frameon = True, bbox_to_anchor=(1.01, 0.3))
# lgnd = ax.legend(loc = "lower right", framealpha = 0.1, frameon = True)

#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [30]

# ======================================================================================================================================================
# Replace columns values by dict
d = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'autumn'}
df["seasons"] = df["seasons"].map(d)

# ======================================================================================================================================================
# Drop rows all equal zero
df.loc[~(df==0).all(axis=1)]

# ======================================================================================================================================================
# First order derivative to time (seconds)
recv.diff() / recv.index.to_series().diff().dt.total_seconds()

# ======================================================================================================================================================
# Long to wide, change multiindex order and  sort by level
dft = dfo.pivot(index='Flux',columns='Tower')
dft.columns.names = ['Info', 'Tower']
dft = dft.T.swaplevel() # or reorder_levels('Info', 'Tower')

idx, iidx = dft.index.sortlevel(0)

# ======================================================================================================================================================
# add legend by scatter size
import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
a2 = 400*np.random.rand(N)

sc = plt.scatter(x, y, s=a2, alpha=0.5)
plt.legend(*sc.legend_elements("sizes", num=6))
plt.show()


# ======================================================================================================================================================
# Group by every 3 minutes
gps = df.groupby(pd.Grouper(freq = '3T'))

for g in gps.groups:
    print(g)
    gp = gps.get_group(g)
    print(gp)

# ======================================================================================================================================================
# Set matplotlib axis datetime format
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d %H:%M')
axes[0].xaxis.set_major_formatter(myFmt)

# ======================================================================================================================================================
# count values
geolocations['Countrycode'].value_counts()



# ======================================================================================================================================================
#  blended transformation
import matplotlib.transforms as transforms
trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

# ======================================================================================================================================================
# convert axes / data transform  
# https://stackoverflow.com/questions/62004022/convert-from-data-coordinates-to-axes-coordinates-in-matplotlib

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# this is in data coordinates
point = (1000, 1000)
trans = ax.transData.transform(point)
trans = ax.transAxes.inverted().transform(trans)
print(ax.get_xlim(), trans)  

ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
trans = ax.transData.transform(point)
trans = ax.transAxes.inverted().transform(trans)
print(ax.get_xlim(), trans)

# =======================================================================================================================
# auto nrows and ncols
acnt = len(df.columns) - 1
nc = int(np.ceil(np.sqrt(acnt)))
if nc*(nc-1) >= acnt:
    nr = nc - 1
else:
    nr = nc

fig, axes = setup_canvas(nr,nc, figsize = (5 * nc, 3 * nr))#, wspace = 0.2, hspace = 0.2)

# =======================================================================================================================
# interp nc to sites
df = nc['o3'].interp(longitude = meta['longitude'].to_xarray(), latitude = meta['latitude'].to_xarray()).to_dataframe()

# =======================================================================================================================
# Smooth 2D array
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import uniform_filter, convolve1d

klen = 2
kernel = np.ones(klen)

arr2Df = arr2D
arr2Df = convolve1d(arr2Df, kernel)
arr2Df = convolve1d(arr2Df.T, kernel).T

nc = nc.coarsen(longitude=2, boundary = 'pad').mean().coarsen(latitude=2, boundary = 'pad').mean()