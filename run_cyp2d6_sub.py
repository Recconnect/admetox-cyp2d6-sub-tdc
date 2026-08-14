"""
TDC CYP2D6_Substrate_CarbonMangels Benchmark — GIN-ContextPred fine-tune (AUPRC)
================================================================================

ADMETox.AI submission for the TDC ADMET Leaderboard (CYP2D6_Substrate_CarbonMangels).

Pipeline:
  Backbone:  gin_supervised_contextpred (Hu et al. 2020, "Strategies for
             Pre-training Graph Neural Networks"), a GIN pretrained on ZINC15
             with ContextPred self-supervision.  This is the exact architecture
             behind the current TDC leaderboard entry (ContextPred, 0.736).
             The pretraining is self-supervised on ZINC15 (no ADMET labels), so
             there is no CYP2D6 label leakage.
  Forward:   gin_pure.py — an exact pure-PyTorch reimplementation of the dgllife
             GIN forward pass (numerically verified against dgllife).  It runs on
             CPU and on AMD ROCm / NVIDIA CUDA GPUs (dgl itself is CPU-only in
             the pinned environment).
  Head:      mean-pooled node representations -> Linear(300 -> 1),
             BCEWithLogitsLoss (pos_weight = neg/pos ~ 3.6).
  Protocol:  train on ALL official train_val (532 molecules); per-seed stratified
             inner validation split (0.15) for early stopping on AUPRC;
             early-stopped best-epoch model predicts the official test set.
             Model is reset to the pretrained checkpoint before every seed, so
             the runs are independent.
  Seeds:     [0..29]  (30 independent runs; TDC requires >= 5)
  Report:    TDC evaluate_many (mean +/- std) + averaged-vector AUPRC

Result:
  TDC evaluate_many:     0.7420 +/- 0.0200  (30 runs)
  Averaged-vector AUPRC: 0.7675
  vs TDC SOTA 0.736 (ContextPred) -> BEAT +0.006/+0.031

Usage:
  python run_cyp2d6_sub.py                          # 30 seeds (full submission)
  python run_cyp2d6_sub.py --seeds 5                # 5 seeds (TDC minimum)
  python run_cyp2d6_sub.py --seeds 0,1,2            # specific seeds
  python run_cyp2d6_sub.py --epochs 60 --balance     # full protocol defaults

Requirements:
  pip install PyTDC --no-deps
  pip install torch dgl dgllife rdkit scikit-learn numpy scipy tqdm
  First run downloads the pretrained checkpoint (~8 MB) from the DGL S3 bucket
  (gin_supervised_contextpred_pre_trained.pth) and, if data/ is empty, the
  official TDC benchmark data.
"""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import average_precision_score

from tdc.benchmark_group import admet_group

from gin_pure import build_pure_model, batch_tensors, mean_pool

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

BENCHMARK_NAME = "CYP2D6_Substrate_CarbonMangels"
SOTA = 0.736  # ContextPred (TDC ADMET leaderboard, scaffold split)
PRETRAIN_KIND = "gin_supervised_contextpred"
HIDDEN = 300

# Full-submission defaults (the verified winning protocol)
SEEDS_DEFAULT = ",".join(str(s) for s in range(30))
EPOCHS_DEFAULT = 60
LR_DEFAULT = 1e-3
BS_DEFAULT = 32
INNER_FRAC_DEFAULT = 0.15


class Head(torch.nn.Module):
    """Classification head on top of the mean-pooled graph representation."""

    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc(self.drop(x)).squeeze(-1)


def load_data():
    """Official TDC scaffold split: 532 train_val / 135 test molecules."""
    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(BENCHMARK_NAME)
    name = benchmark["name"]
    train_val = benchmark["train_val"]
    test = benchmark["test"]
    tv_smiles = list(train_val["Drug"].values)
    te_smiles = list(test["Drug"].values)
    y_tv = train_val["Y"].values.astype(int)
    y_te = test["Y"].values.astype(int)
    return tv_smiles, te_smiles, y_tv, y_te, group, name


def build_graphs(smiles_list):
    """SMILES -> per-graph tensors using dgllife's pretrain featurizers."""
    from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
    atom_f = PretrainAtomFeaturizer()
    bond_f = PretrainBondFeaturizer(self_loop=True)
    graphs = []
    n_ok = 0
    for sm in smiles_list:
        try:
            g = smiles_to_bigraph(sm, add_self_loop=True, node_featurizer=atom_f,
                                  edge_featurizer=bond_f)
        except Exception:
            g = None
        if g is None or g.number_of_nodes() == 0:
            g = _fallback_graph()
        else:
            n_ok += 1
        src, dst = g.edges()
        graphs.append({
            "atom_id": g.ndata["atomic_number"],
            "chir": g.ndata["chirality_type"],
            "bond_type": g.edata["bond_type"],
            "bond_dir": g.edata["bond_direction_type"],
            "src": src, "dst": dst,
            "n_nodes": int(g.number_of_nodes()),
        })
    print(f"  graphs built: {len(graphs)} ({n_ok} parsed via RDKit)", flush=True)
    return graphs


def _fallback_graph():
    import dgl
    g = dgl.graph(([], []))
    g.add_nodes(1)
    g.add_edges([0], [0])
    g.ndata["atomic_number"] = torch.tensor([6], dtype=torch.long)
    g.ndata["chirality_type"] = torch.tensor([0], dtype=torch.long)
    g.edata["bond_type"] = torch.tensor([0], dtype=torch.long)
    g.edata["bond_direction_type"] = torch.tensor([0], dtype=torch.long)
    return g


def build_pretrained(device):
    """Pretrained GIN-ContextPred -> pure-PyTorch model (first call downloads)."""
    from dgllife.model import load_pretrained
    dgl_model = load_pretrained(PRETRAIN_KIND)
    model = build_pure_model(dgl_model, device=device)
    del dgl_model
    head = Head(HIDDEN).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in head.parameters())
    n_tot = sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"  loaded {PRETRAIN_KIND}: {n_train/1e6:.2f}M trainable / "
          f"{n_tot/1e6:.2f}M total params", flush=True)
    return model, head


def predict(model, head, graphs, indices, device):
    """Sigmoid probabilities for graphs[indices] (batched inference)."""
    model.eval()
    head.eval()
    probs = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            ids = indices[i:i + bs]
            info = batch_tensors(graphs, ids, device)
            h = model([info["atom_id"], info["chir"]], info["src"], info["dst"],
                      [info["bond_type"], info["bond_dir"]])
            pool = mean_pool(h, info)
            probs.append(torch.sigmoid(head(pool)).cpu().numpy())
    return np.concatenate(probs)


def fit_seed(model, head, graphs, tr_idx, y_tv, te_idx, y_te, args, device, seed):
    """Fine-tune with stratified inner-val early stopping on AUPRC.

    tr_idx/te_idx index into the combined (train_val + test) label array.
    Returns sigmoid probabilities for te_idx.
    """
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    np.random.seed(seed)

    idx_tr2, idx_iv = train_test_split(tr_idx, test_size=args.inner_frac,
                                       stratify=y_tv[tr_idx], random_state=seed * 7919 + 1)
    y_iv = y_tv[idx_iv]

    pos_w = None
    if args.balance:
        pos = int(y_tv[idx_tr2].sum())
        neg = int(len(idx_tr2) - pos)
        pos_w = torch.tensor(neg / max(pos, 1), device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=args.wd)
    total_steps = int(np.ceil(len(idx_tr2) / args.bs)) * args.epochs
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm))

    t0 = time.time()
    best_auprc, best_model, best_head = -1.0, None, None
    for ep in range(args.epochs):
        model.train()
        head.train()
        order = np.random.permutation(len(idx_tr2))
        for start in range(0, len(order), args.bs):
            ids = [idx_tr2[k] for k in order[start:start + args.bs]]
            info = batch_tensors(graphs, ids, device)
            h = model([info["atom_id"], info["chir"]], info["src"], info["dst"],
                      [info["bond_type"], info["bond_dir"]])
            pool = mean_pool(h, info)
            loss = crit(head(pool),
                        torch.tensor(y_tv[ids].astype(np.float32), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        p_iv = predict(model, head, graphs, list(idx_iv), device)
        auprc_iv = float(average_precision_score(y_iv, p_iv))
        if auprc_iv > best_auprc:
            best_auprc = auprc_iv
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        print(f"    [seed {seed}] ep {ep+1}/{args.epochs} loss={loss.item():.4f} "
              f"inner_auprc={auprc_iv:.4f} best={best_auprc:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(best_model)
    head.load_state_dict(best_head)
    return predict(model, head, graphs, list(te_idx), device)


def main():
    parser = argparse.ArgumentParser(
        description="TDC CYP2D6_Substrate_CarbonMangels — GIN-ContextPred fine-tune (AUPRC)")
    parser.add_argument("--seeds", type=str, default=SEEDS_DEFAULT,
                        help="Number of seeds (e.g. 5 -> seeds 0..4) or comma-separated "
                             "list. Default: 30")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--bs", type=int, default=BS_DEFAULT)
    parser.add_argument("--inner-frac", type=float, default=INNER_FRAC_DEFAULT)
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--wd", type=float, default=0.0)
    args = parser.parse_args()

    parts = args.seeds.split(",")
    if len(parts) == 1 and parts[0].isdigit():
        seeds = list(range(int(parts[0])))
    else:
        seeds = [int(s) for s in parts]

    t_start = time.time()
    print("=" * 60)
    print("TDC CYP2D6_Substrate_CarbonMangels — GIN-ContextPred Fine-tune")
    print("  ADMETox.AI")
    print("=" * 60)
    print(f"  Backbone:  gin_supervised_contextpred (ZINC15-pretrained GIN, Hu 2020)")
    print(f"  Protocol:  train on all train_val, stratified inner-val early stop (AUPRC)")
    print(f"  Hyper:     ep={args.epochs} lr={args.lr} bs={args.bs} "
          f"inner={args.inner_frac} balance={args.balance}")
    print(f"  Seeds:     {seeds}  ({len(seeds)} runs)")
    print(f"  SOTA to beat: {SOTA} (ContextPred, 2026)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    tv_smiles, te_smiles, y_tv, y_te, group, name = load_data()
    print(f"  Train+Val: {len(tv_smiles)}, Test: {len(te_smiles)}", flush=True)

    graphs = build_graphs(tv_smiles + te_smiles)
    g_tv, g_te = graphs[:len(tv_smiles)], graphs[len(tv_smiles):]
    model, head = build_pretrained(device)
    init_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

    tr_idx = np.arange(len(tv_smiles))
    te_idx = np.arange(len(tv_smiles), len(tv_smiles) + len(te_smiles))
    preds = np.zeros((len(seeds), len(te_smiles)), dtype=np.float64)
    auprcs = []
    for si, seed in enumerate(seeds):
        model.load_state_dict(init_model)
        head.load_state_dict(init_head)
        p = fit_seed(model, head, graphs, tr_idx, y_tv, te_idx, y_te,
                     args, device, seed)
        preds[si] = p
        auprcs.append(float(average_precision_score(y_te, p)))
        tag = "BEAT" if auprcs[-1] > SOTA else " "
        print(f"  Seed {seed:2d}: AUPRC = {auprcs[-1]:.4f} {tag} "
              f"[{time.time()-t_start:.0f}s elapsed]", flush=True)

    ens_vec = preds.mean(axis=0)
    ens_auprc = float(average_precision_score(y_te, ens_vec))

    if len(preds) >= 5:
        tdc_raw = group.evaluate_many([{name: p} for p in preds])
        tdc_list = tdc_raw[name]
        tdc_val, tdc_std = float(tdc_list[0]), float(tdc_list[1])
    else:
        tdc_raw = {}
        tdc_val, tdc_std = float("nan"), float("nan")

    tdc_vec = float(list(list(group.evaluate({name: ens_vec}).values())[0].values())[0])

    mean, std = float(np.mean(auprcs)), float(np.std(auprcs))
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  SOTA:                   {SOTA} (ContextPred)")
    print(f"  Mean +/- Std:           {mean:.4f} +/- {std:.4f}")
    print(f"  TDC evaluate_many:      {tdc_val:.4f} +/- {tdc_std:.4f}  ({len(seeds)} runs)")
    print(f"  Averaged-vector AUPRC:  {ens_auprc:.4f}")
    print(f"  TDC evaluate (vector):  {tdc_vec:.4f}")
    if mean > SOTA:
        print(f"\n  *** BEAT SOTA by +{mean - SOTA:.4f} (mean) / +{ens_auprc - SOTA:.4f} (vector) ***")
    else:
        print(f"\n  Gap to SOTA: {mean - SOTA:.4f}")
    print(f"  Time: {(time.time()-t_start)/60:.1f} min")
    print("=" * 60)

    result = {
        "name": name,
        "protocol": f"gin_contextpred_ft_ep{args.epochs}_lr{args.lr}_bs{args.bs}"
                    f"_inner{args.inner_frac}_bal{int(args.balance)}_n{len(seeds)}",
        "tdc_results": {k: list(v) if isinstance(v, (list, tuple)) else str(v)
                        for k, v in tdc_raw.items()},
        "mean": mean, "std": std,
        "auprcs": [float(a) for a in auprcs],
        "ensemble_vector_auprc": ens_auprc,
        "tdc_evaluate_vector": tdc_vec,
        "seeds": seeds,
        "model": PRETRAIN_KIND,
        "sota": SOTA, "beat_sota": bool(mean > SOTA),
        "gap": float(mean - SOTA),
        "runtime_seconds": round(time.time() - t_start, 1),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_file = OUTPUT_DIR / "cyp2d6_sub_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(OUTPUT_DIR / "cyp2d6_sub_preds.npz",
                        y=y_te, preds=preds, seeds=np.asarray(seeds))
    with open(OUTPUT_DIR / "predictions_test.txt", "w") as f:
        f.write("\n".join(f"{v:.6f}" for v in ens_vec) + "\n")
    print(f"\n  Saved: {out_file} + cyp2d6_sub_preds.npz + predictions_test.txt")


if __name__ == "__main__":
    main()
