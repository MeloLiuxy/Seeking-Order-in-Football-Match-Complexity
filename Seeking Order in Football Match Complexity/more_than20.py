# -*- coding: utf-8 -*-
import ast
import numpy as np
import pandas as pd
import os

INPUT_XLS  = r""
SHEET_NAME = 0
OUTPUT_XLS = r""

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

def _frame_count(v):
    """统计帧中的元素个数；非列表/缺失返回 0。"""
    v = _maybe_eval(v)
    if isinstance(v, list):
        return len(v)
    return 0

def main():
    df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)

    # 确保列存在（没有就当空列）
    for c in ["freeze_frame", "back_ff"]:
        if c not in df.columns:
            df[c] = np.nan

    # 统计人数
    df["freeze_frame_count"] = df["freeze_frame"].apply(_frame_count)
    df["back_ff_count"]      = df["back_ff"].apply(_frame_count)

    # 筛选条件：两者都 > 18
    mask = (df["freeze_frame_count"] > 20) & (df["back_ff_count"] > 20)
    filt = df.loc[mask].copy()

    # 输出
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLS)), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as w:
        filt.to_excel(w, index=False, sheet_name="over18_both")

    print(f"[OK] 原始行数: {len(df)}")
    print(f"[OK] 满足 freeze_frame>20 且 back_ff>20 的行数: {len(filt)}")
    print(f"[OK] 已写出: {OUTPUT_XLS}")

if __name__ == "__main__":
    main()
