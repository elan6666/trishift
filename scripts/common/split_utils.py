from __future__ import annotations

import random
import numpy as np
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


def split_by_dataset_policy(data, dataset_name: str, seed: int, test_ratio: float | None = None):
    """Rebuild dataset splits with the same policy used by the main runners."""
    name = str(dataset_name).strip().lower()
    if name == "norman":
        label_key = getattr(data, "label_key", "condition")
        adata_all = getattr(data, "adata_all")
        subgroup_df = norman_subgroup(
            list(adata_all.obs[label_key].astype(str).unique()),
            seed=int(seed),
        )
        test_conds = list(subgroup_df[subgroup_df.group == "test"].index)
        val_conds = list(subgroup_df[subgroup_df.group == "val"].index)
        return data.split_by_condition(
            seed=int(seed),
            test_conds=test_conds,
            val_conds=val_conds,
        )

    if test_ratio is None:
        return data.split_by_condition(seed=int(seed))
    return data.split_by_condition(seed=int(seed), test_ratio=float(test_ratio))


def _split_conditions_for_holdout(
    values: list[str],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    vals = np.asarray([str(x) for x in values], dtype=object)
    if vals.size == 0:
        return [], []
    rng = np.random.RandomState(int(seed))
    rng.shuffle(vals)
    n_test = int(round(float(ratio) * int(vals.size)))
    if float(ratio) > 0.0 and int(vals.size) > 1 and n_test == 0:
        n_test = 1
    if n_test >= int(vals.size):
        n_test = max(1, int(vals.size) - 1)
    test_arr, train_arr = np.split(vals, [n_test])
    return [str(x) for x in test_arr.tolist()], [str(x) for x in train_arr.tolist()]


def _split_ctrl_indices_train_val_test(
    *,
    adata,
    label_key: str,
    ctrl_label: str,
    seed: int,
    test_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cond_series = adata.obs[label_key].astype(str)
    ctrl_idx = np.where(cond_series.values == str(ctrl_label))[0]
    if ctrl_idx.size < 3:
        raise ValueError(
            "unseen_ctrl_unseen_perturbation requires at least 3 ctrl cells; "
            f"found {ctrl_idx.size}"
        )
    rng = np.random.RandomState(int(seed) + 20011)
    perm = rng.permutation(ctrl_idx)

    n_test = int(round(float(test_ratio) * int(ctrl_idx.size)))
    n_test = min(max(n_test, 1), int(ctrl_idx.size) - 2)
    test_idx = perm[:n_test]
    remaining = perm[n_test:]

    n_val = int(round(float(val_ratio) * int(remaining.size)))
    n_val = min(max(n_val, 1), int(remaining.size) - 1)
    val_idx = remaining[:n_val]
    train_idx = remaining[n_val:]
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "unseen_ctrl_unseen_perturbation produced an empty ctrl split: "
            f"train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
        )
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


def _condition_split_for_dataset(
    data,
    dataset_name: str,
    *,
    seed: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str], pd.DataFrame | None]:
    """Return the perturbation split used by the default benchmark policy.

    Unseen-control evaluation must compare against the same perturbation holdout
    as the default benchmark; only the control-cell source changes.  Therefore
    this helper delegates to ``split_by_dataset_policy`` instead of reimplementing
    a separate random perturbation split.
    """
    name = str(dataset_name).strip().lower()
    ctrl_label = getattr(data, "ctrl_label", "ctrl")
    default_split = split_by_dataset_policy(
        data,
        dataset_name=name,
        seed=int(seed),
        test_ratio=float(test_ratio),
    )
    train_conds = [str(c) for c in default_split.get("train_conds", []) if str(c) != ctrl_label]
    val_conds = [str(c) for c in default_split.get("val_conds", []) if str(c) != ctrl_label]
    test_conds = [str(c) for c in default_split.get("test_conds", []) if str(c) != ctrl_label]
    if name == "norman":
        label_key = getattr(data, "label_key", "condition")
        adata_all = getattr(data, "adata_all")
        subgroup_df = norman_subgroup(
            list(adata_all.obs[label_key].astype(str).unique()),
            seed=int(seed),
        )
        return train_conds, val_conds, test_conds, subgroup_df

    return train_conds, val_conds, test_conds, None


def split_unseen_ctrl_unseen_perturbation(
    data,
    dataset_name: str,
    *,
    seed: int,
    test_ratio: float = 0.2,
    pert_val_ratio: float = 0.1,
    ctrl_test_ratio: float = 0.2,
    ctrl_val_ratio: float = 0.1,
) -> tuple[dict, pd.DataFrame | None]:
    """Build one split with disjoint ctrl cells and unseen perturbation conditions.

    The returned split is intended to be used by training and evaluation together:
    Stage/training code sees train/val ctrl only, while evaluation uses test ctrl.
    """
    adata = getattr(data, "adata_all")
    label_key = getattr(data, "label_key", "condition")
    ctrl_label = getattr(data, "ctrl_label", "ctrl")

    train_conds, val_conds, test_conds, subgroup_df = _condition_split_for_dataset(
        data,
        dataset_name,
        seed=int(seed),
        test_ratio=float(test_ratio),
        val_ratio=float(pert_val_ratio),
    )
    if not train_conds or not val_conds or not test_conds:
        raise ValueError(
            "unseen_ctrl_unseen_perturbation produced an empty perturbation split: "
            f"train={len(train_conds)}, val={len(val_conds)}, test={len(test_conds)}"
        )

    train_ctrl_idx, val_ctrl_idx, test_ctrl_idx = _split_ctrl_indices_train_val_test(
        adata=adata,
        label_key=label_key,
        ctrl_label=ctrl_label,
        seed=int(seed),
        test_ratio=float(ctrl_test_ratio),
        val_ratio=float(ctrl_val_ratio),
    )

    cond_series = adata.obs[label_key].astype(str)
    train_mask = cond_series.isin(train_conds).values
    val_mask = cond_series.isin(val_conds).values
    test_mask = cond_series.isin(test_conds).values

    train_ctrl_mask = np.zeros(adata.n_obs, dtype=bool)
    val_ctrl_mask = np.zeros(adata.n_obs, dtype=bool)
    test_ctrl_mask = np.zeros(adata.n_obs, dtype=bool)
    train_ctrl_mask[train_ctrl_idx] = True
    val_ctrl_mask[val_ctrl_idx] = True
    test_ctrl_mask[test_ctrl_idx] = True

    train_mask = train_mask | train_ctrl_mask
    val_mask = val_mask | val_ctrl_mask
    test_mask = test_mask | test_ctrl_mask
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not np.any(mask):
            raise ValueError(f"unseen_ctrl_unseen_perturbation produced empty {split_name} set")

    split_dict = {
        "train": adata[train_mask],
        "val": adata[val_mask],
        "test": adata[test_mask],
        "train_conds": [str(c) for c in train_conds],
        "val_conds": [str(c) for c in val_conds],
        "test_conds": [str(c) for c in test_conds],
        "split_policy": "unseen_ctrl_unseen_perturbation",
        "ctrl_split_policy": "train_val_test",
        "ctrl_test_ratio": float(ctrl_test_ratio),
        "ctrl_val_ratio": float(ctrl_val_ratio),
        "train_ctrl_n": int(train_ctrl_idx.size),
        "val_ctrl_n": int(val_ctrl_idx.size),
        "test_ctrl_n": int(test_ctrl_idx.size),
    }
    return split_dict, subgroup_df
