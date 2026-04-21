# -*- coding: utf-8 -*-
import ast
import os
import numpy as np
import pandas as pd

# 要处理的文件：输入 -> 输出
FILE_PAIRS = [
    (
        r"",
        r""
    ),
    (
        r"",
        r""
    ),
]

SHEET_NAME  = 0

# 进攻方向：+x 表示从左到右；-x 表示从右到左
ATTACK_POSX = True  # 若为从右到左，请改为 False

# 判定阈值（避免极小数值抖动），单位：米
EPS = 1e-6

def _maybe_eval(x):
    """把字符串形式的 list/dict 安全转对象；否则原样返回。"""
    if isinstance(x, (list, dict, tuple)) or pd.isna(x):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x

def _xy_from_loc(v):
    """location 列若是 [x,y] 列表则拆出 (x,y)，否则返回 (nan,nan)。"""
    v = _maybe_eval(v)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan

def ensure_xy_cols(df):
    """确保存在 location_x/location_y 与 end_location_x/end_location_y 列。"""
    df = df.copy()

    if "location_x" not in df.columns or "location_y" not in df.columns:
        if "location" in df.columns:
            xy = df["location"].apply(_xy_from_loc)
            df["location_x"] = xy.apply(lambda t: t[0])
            df["location_y"] = xy.apply(lambda t: t[1])
        else:
            df["location_x"] = np.nan
            df["location_y"] = np.nan

    if "end_location_x" not in df.columns or "end_location_y" not in df.columns:
        # 有些表 end_location 可能是列表；若没有就保持 NaN
        if "end_location" in df.columns:
            exy = df["end_location"].apply(_xy_from_loc)
            df["end_location_x"] = exy.apply(lambda t: t[0])
            df["end_location_y"] = exy.apply(lambda t: t[1])
        else:
            df.setdefault("end_location_x", np.nan)
            df.setdefault("end_location_y", np.nan)

    return df

def process_one_file(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"[WARN] 找不到输入文件：{in_path}")
        return

    df = pd.read_excel(in_path, sheet_name=SHEET_NAME)

    # 确保有必要的坐标列
    df = ensure_xy_cols(df)

    # 计算位移
    df["dx"] = df["end_location_x"] - df["location_x"]
    df["dy"] = df["end_location_y"] - df["location_y"]

    # 向前判定：若进攻方向为 +x，则 dx > EPS；若为 -x，则 dx < -EPS
    if ATTACK_POSX:
        mask_forward = df["dx"] > EPS
    else:
        mask_forward = df["dx"] < -EPS

    forward_df = df.loc[mask_forward].copy()

    # 输出
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        forward_df.to_excel(w, index=False, sheet_name="forward_only")

    print(f"[OK] 文件: {in_path}")
    print(f"[OK] 原始行数: {len(df)}")
    print(f"[OK] 向前事件行数: {len(forward_df)} (ATTACK_POSX={ATTACK_POSX})")
    print(f"[OK] 已写出: {out_path}")
    print("-" * 60)

def main():
    for in_path, out_path in FILE_PAIRS:
        process_one_file(in_path, out_path)

if __name__ == "__main__":
    main()
