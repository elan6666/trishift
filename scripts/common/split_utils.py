from __future__ import annotations

import random
import pandas as pd


def condition_sort(x: str) -> str:
    return "+".join(sorted(str(x).split("+")))


def split_list_by_ratio(lst: list[str], ratios: list[float], seed: int) -> list[list[str]]:
    work = list(lst)
    random.seed(seed)
    random.shuffle(work)

    total_ratio = sum(ratios)
    total_length = len(work)
    num_splits = len(ratios)
    if total_length < num_splits:
        raise ValueError("The length of the list must be >= number of ratios")

    sizes = [max(1, int(total_length * ratio / total_ratio)) for ratio in ratios]
    while sum(sizes) > total_length:
        sizes[sizes.index(max(sizes))] -= 1
    sizes[-1] += total_length - sum(sizes)

    out: list[list[str]] = []
    cur = 0
    for size in sizes:
        out.append(work[cur : cur + size])
        cur += size
    return out


def norman_subgroup(pert_list: list[str], seed: int) -> pd.DataFrame:
    uniq_perts = list(pert_list)
    uniq_perts.remove("ctrl")
    uniq_combos = [p for p in uniq_perts if "ctrl" not in p.split("+")]
    uniq_singles = [p for p in uniq_perts if "ctrl" in p.split("+")]

    test_single, val_single, train_single = split_list_by_ratio(
        uniq_singles, [0.2, 0.05, 0.75], seed=seed
    )

    combo_seen0 = [
        p for p in uniq_combos if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 0
    ]
    combo_seen1 = [
        p for p in uniq_combos if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 1
    ]
    combo_seen2 = [
        p for p in uniq_combos if sum([i + "+ctrl" in train_single for i in p.split("+")]) == 2
    ]

    test_seen0, val_seen0, train_seen0 = split_list_by_ratio(combo_seen0, [0.2, 0.1, 0.7], seed=seed)
    test_seen1, val_seen1, train_seen1 = split_list_by_ratio(combo_seen1, [0.2, 0.1, 0.7], seed=seed)
    test_seen2, val_seen2, train_seen2 = split_list_by_ratio(combo_seen2, [0.2, 0.1, 0.7], seed=seed)

    test_all = test_single + test_seen0 + test_seen1 + test_seen2
    val_all = val_single + val_seen0 + val_seen1 + val_seen2
    train_all = train_single + train_seen0 + train_seen1 + train_seen2

    group = [
        "train" if p in train_all else "val" if p in val_all else "test"
        for p in uniq_perts
    ]
    subgroup_list: list[str] = []
    for p in uniq_perts:
        if p in uniq_singles:
            subgroup_list.append("single")
        elif p in combo_seen0:
            subgroup_list.append("seen0")
        elif p in combo_seen1:
            subgroup_list.append("seen1")
        elif p in combo_seen2:
            subgroup_list.append("seen2")
        else:
            subgroup_list.append("unknown")

    return pd.DataFrame({"group": group, "subgroup": subgroup_list}, index=uniq_perts)
