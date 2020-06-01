from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def main():
    paths = Path(".").glob(r"*xsites*")
    for p in paths:
        code_names = []
        r2_list = []
        std_list = []
        print(p.stem)
        out_path = p.joinpath("figs")
        if not out_path.exists(): out_path.mkdir()
        res_paths = p.joinpath("1data4pics").glob(r"*A*")
        for rp in res_paths:
            print(rp.stem)
            df = pd.read_csv(rp)
            slope, intercept, r_value, p_value, std_err = stats.linregress(df["pred"], df["ys"])
            code_names.append(rp.stem)
            r2_list.append(r_value)
            std_list.append(std_err)
        df = pd.DataFrame({
            "code": code_names,
            "r2": r2_list,
            "std": std_list
        })
        df.to_csv(p.stem + ".csv")
        



if __name__ == "__main__":
    main()