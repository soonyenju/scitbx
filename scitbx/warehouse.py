# sort path by time
path_time = [pd.to_datetime("".join(p.stem.split("_")[-4::]), format = "%Y%m%d%H%M") for p in paths]
# [x for _, x in sorted(zip(Y, X))] # sort X by Y values
paths = [p for _, p in sorted(zip(path_time, paths))]


if float(lat) >= 0:
    dfc["seasons"] = (dfc.index.month%12 + 3)//3
else:
    dfc["seasons"] = ((dfc.index.month + 6)%12 + 3)//3