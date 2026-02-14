import ast
from pathlib import Path
import hashlib
import random
import sys

import numpy as np
import pandas as pd
import anndata as ad
import torch
import yaml

sys.path.insert(0, str(Path("src").resolve()))

from trishift.TriShiftData import TriShiftData
from trishift.TriShift import TriShift
from trishift import _utils
from trishift._model import aggregate_cond_embedding


def _make_bad_list():
    parts = [("sc", "outer"), ("sc", "pram"), ("sys", "tema"), ("gene", "pt")]
    return [a + b for a, b in parts]


def _fail(msg: str):
    print(msg)
    raise SystemExit(1)


def _check_banned():
    bad = _make_bad_list()
    roots = [Path("src") / "trishift", Path("scripts")]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                low = line.lower()
                for w in bad:
                    if w in low:
                        hits.append((path, i, line))
                        break
    if hits:
        print("Banned scan: FAIL")
        for path, ln, line in hits:
            print(f"{path}:{ln}: {line}")
        raise SystemExit(1)
    print("Banned scan: PASS")


def _check_hash_usage():
    path = Path("src") / "trishift" / "TriShift.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = False

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            nonlocal found
            if node.name != "TriShift":
                return
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "evaluate":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                            if sub.func.id == "hash":
                                found = True

    _Visitor().visit(tree)
    if found:
        print("Seed stability: FAIL (built-in hash used)")
        raise SystemExit(1)
    print("Seed stability: PASS")


def _check_topk_cache_meta():
    path = Path("src") / "trishift" / "TriShiftData.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
                if node.func.attr == "savez":
                    for kw in node.keywords:
                        if kw.arg is not None:
                            keys.add(kw.arg)
    need = {"mode", "k", "seed", "candidates"}
    if need.issubset(keys):
        print("TopK cache meta: PASS")
    else:
        missing = sorted(list(need - keys))
        print(f"TopK cache meta: WARN missing {missing}")
        print("Suggested key: mode_k_seed_candidates")


def _flat_params(mod):
    params = [p.detach().cpu().flatten() for p in mod.parameters()]
    if not params:
        return torch.tensor([])
    return torch.cat(params)


def _make_tiny_data():
    rng = np.random.default_rng(0)
    n_ctrl = 20
    n_per = 12
    n_genes = 40
    X_ctrl = rng.random((n_ctrl, n_genes), dtype=np.float32)
    X_a = rng.random((n_per, n_genes), dtype=np.float32)
    X_b = rng.random((n_per, n_genes), dtype=np.float32)
    X_ab = rng.random((n_per, n_genes), dtype=np.float32)
    X = np.vstack([X_ctrl, X_a, X_b, X_ab])
    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * n_ctrl
            + ["A"] * n_per
            + ["B"] * n_per
            + ["A+B"] * n_per
        }
    )
    gene_names = ["A", "B"] + [f"G{i}" for i in range(n_genes - 2)]
    var = pd.DataFrame({"gene_name": gene_names})
    adata = ad.AnnData(X=X, obs=obs, var=var)
    top20 = np.arange(20, dtype=int)
    adata.uns["top20_degs_non_dropout"] = {
        "A": top20.copy(),
        "B": top20.copy(),
        "A+B": top20.copy(),
    }
    embd_df = pd.DataFrame(
        rng.random((3, 8), dtype=np.float32),
        index=["ctrl", "A", "B"],
    )
    return adata, embd_df


def _dynamic_checks():
    adata, embd_df = _make_tiny_data()
    data = TriShiftData(adata, embd_df)
    data.setup_embedding_index()
    data.build_or_load_degs()

    for cond in data.conditions_pert:
        degs = data.adata_all.uns["top20_degs_final"][cond]
        drop = set(data.cond_to_gene_idx.get(cond, []))
        if any(d in drop for d in degs):
            _fail("DE gene removal: FAIL")

    tri = TriShift(data, device="cpu")
    tri.model_init(
        x_dim=adata.n_vars,
        z_dim=8,
        cond_dim=8,
        vae_enc_hidden=[16],
        vae_dec_hidden=[16],
        shift_hidden=[16],
        gen_hidden=[16],
        dropout=0.0,
    )

    vae_before = _flat_params(tri.net.vae).clone()
    shift_before = _flat_params(tri.net.shift).clone()
    gen_before = _flat_params(tri.net.gen).clone()

    tri.train_stage1_vae(
        data.adata_ctrl,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
    )

    vae_after = _flat_params(tri.net.vae)
    shift_after = _flat_params(tri.net.shift)
    gen_after = _flat_params(tri.net.gen)

    if torch.allclose(vae_before, vae_after):
        _fail("Stage1 only VAE: FAIL (vae unchanged)")
    if not torch.allclose(shift_before, shift_after):
        _fail("Stage1 only VAE: FAIL (shift changed)")
    if not torch.allclose(gen_before, gen_after):
        _fail("Stage1 only VAE: FAIL (gen changed)")
    print("Stage1 only VAE: PASS")

    tri.encode_and_cache_mu(data.adata_all, batch_size=16, amp=False, num_workers=0, pin_memory=False)

    vae_frozen = _flat_params(tri.net.vae).clone()

    splits = data.split_by_condition(seed=1, test_ratio=0.5, val_ratio=0.0)
    emb_table = torch.randn(3, 8)

    import trishift.TriShift as ts_mod

    real_dl = ts_mod.DataLoader
    captured = {}

    class _DLSpy:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            self._dl = real_dl(dataset, **kwargs)

        def __iter__(self):
            return iter(self._dl)

        def __len__(self):
            return len(self._dl)

    ts_mod.DataLoader = _DLSpy
    try:
        tri.train_stage23_joint(
            splits,
            emb_table,
            mode="knn",
            k=3,
            split_id=1,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            amp=False,
            num_workers=0,
            pin_memory=False,
            grad_accum_steps=1,
        )
    finally:
        ts_mod.DataLoader = real_dl

    if not torch.allclose(vae_frozen, _flat_params(tri.net.vae)):
        _fail("Stage1 frozen: FAIL (vae changed)")
    print("Stage1 frozen: PASS")

    ds = captured.get("dataset", None)
    if ds is None:
        _fail("TopK sampling: FAIL (dataset capture)")

    topk = ds.topk
    ctrl_X = ds.ctrl_X
    idxs = []
    for _ in range(50):
        sample = ds[0]
        ctrl_expr = sample[0]
        matches = np.where(np.all(np.isclose(ctrl_X, ctrl_expr, atol=1e-6), axis=1))[0]
        if matches.size == 0:
            _fail("TopK sampling: FAIL (ctrl not found)")
        idxs.append(int(matches[0]))

    if not all(i in set(topk[0]) for i in idxs):
        _fail("TopK sampling: FAIL (out of topk)")
    if len(set(idxs)) <= 1:
        _fail("TopK sampling: FAIL (no randomness)")

    x_ctrl_batch = torch.tensor(np.stack([ds[i][0] for i in range(4)]), dtype=torch.float32)
    cond_vec = torch.stack(
        [aggregate_cond_embedding(emb_table, ds[i][2], mode="sum") for i in range(4)],
        dim=0,
    )
    out = tri.net.forward_joint(x_ctrl_batch, cond_vec)
    mu_direct = tri.net.vae.encode_mu(x_ctrl_batch)
    if not torch.allclose(out["z_ctrl_mu"], mu_direct):
        _fail("TopK sampling: FAIL (stage share mismatch)")

    print("TopK sampling: PASS")
    print(f"TopK example: {topk[0].tolist()} -> {sorted(set(idxs))[:5]}")

    defaults = _utils.load_yaml("configs/defaults.yaml")
    base_seed = int(defaults.get("seed", 24))

    import trishift._utils as utils_mod

    real_sampler = utils_mod.sample_control_indices
    records = {}
    conds = splits["test_conds"]
    idx_ref = {"i": 0}

    def _sampler(n_ctrl, size, seed):
        cond = conds[idx_ref["i"]]
        arr = real_sampler(n_ctrl, size, seed)
        records[cond] = arr
        idx_ref["i"] += 1
        return arr

    utils_mod.sample_control_indices = _sampler
    try:
        df1 = tri.evaluate(splits, emb_table, split_id=2, n_ensemble=300, base_seed=base_seed)
    finally:
        utils_mod.sample_control_indices = real_sampler

    if df1.shape[0] == 0:
        _fail("Eval ensemble: FAIL (no rows)")

    if len(records) == 0:
        _fail("Eval ensemble: FAIL (no samples)")

    for cond, arr in records.items():
        if arr.shape[0] != 300:
            _fail("Eval ensemble: FAIL (size)")
        uniq = np.unique(arr)
        if uniq.size >= 300:
            _fail("Eval ensemble: FAIL (no replacement)")
        print(f"Eval cond {cond}: uniq={uniq.size}, seed={base_seed+2}, head={arr[:10].tolist()}")

    if len(records) >= 2:
        cond_list = list(records.keys())
        if np.array_equal(records[cond_list[0]], records[cond_list[1]]):
            _fail("Eval ensemble: FAIL (not independent)")

    records2 = {}
    idx_ref["i"] = 0

    def _sampler2(n_ctrl, size, seed):
        cond = conds[idx_ref["i"]]
        arr = real_sampler(n_ctrl, size, seed)
        records2[cond] = arr
        idx_ref["i"] += 1
        return arr

    utils_mod.sample_control_indices = _sampler2
    try:
        _ = tri.evaluate(splits, emb_table, split_id=2, n_ensemble=300, base_seed=base_seed)
    finally:
        utils_mod.sample_control_indices = real_sampler

    for cond in records:
        if not np.array_equal(records[cond], records2.get(cond)):
            _fail("Eval ensemble: FAIL (not reproducible)")

    print("Eval ensemble: PASS")


if __name__ == "__main__":
    _check_banned()
    _check_hash_usage()
    _check_topk_cache_meta()
    _dynamic_checks()
    print("ALL PASS")
