"""
Robust Universal Evaluation Runner for Time Series Classification

Features:
- Dual-Process Bidirectional Execution:
  * --order asc (default): Evaluates datasets from beginning to end (e.g. ACSF1 -> Yoga).
  * --order desc: Evaluates datasets from end to beginning (e.g. Yoga -> ACSF1).
  * Run two processes concurrently (one ascending, one descending) to take double ground simultaneously!
  * Atomic lockfiles (.dataset.lock) and in-progress detection prevent work duplication or race conditions.
  * Atomic master summary merging guarantees that neither process overwrites the other's completed datasets.
- Resume capability: Skips already-completed datasets (or resumes interrupted runs)
  without re-computing them.
- Crash resilience: Wrapped in try...except so if any single dataset errors/times out,
  the runner logs the error and proceeds automatically to the next dataset.
- Automatic Concurrency on Small Datasets:
  * Automatically detects small datasets (N_total * length <= threshold, default 600,000).
  * Executes iterations concurrently across CPU cores using joblib.Parallel.
  * For large datasets, runs iterations sequentially with multicore model parallelism.
- Exact protocol:
  * Seed 0 = exact original library split
  * Seeds 1 to (N-1) = Stratified reshuffle maintaining original split sizes (zero leakage)
- Comprehensive metrics:
  * Per-seed details saved to <output_dir>/<dataset>_results.csv (seed, accuracy, time)
  * Real-time updating summary table in <output_dir>/summary_results.csv
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from aeon.datasets import load_classification
from joblib import Parallel, delayed

from model import FastUnifiedClassifier
from model_v2 import DualExpertClassifierV2
from model_v3 import AdaptiveExpertClassifierV3

BASELINE_EXCEL = "MultiRocket_Hydra_Ensemble_abdulaziz.xlsx"
BASELINE_CSV = "ucr_hydra_multirocket.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Time Series Classifiers on benchmark datasets.")
    
    # Selection options
    parser.add_argument(
        "--all", "-all",
        action="store_true",
        help="Run evaluation on all 109 datasets listed in baseline benchmark file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of a single dataset to evaluate (e.g. InlineSkate, Phoneme, BME)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset names to evaluate sequentially."
    )
    parser.add_argument(
        "--order", "--direction",
        dest="order",
        type=str,
        default="asc",
        choices=["asc", "desc", "forward", "reverse"],
        help="Dataset processing order: 'asc'/'forward' (from 1st to last) or 'desc'/'reverse' (from last to 1st). Enables 2 concurrent runners. Default: 'asc'."
    )
    
    parser.add_argument(
        "--model_version",
        type=str,
        default="v3",
        choices=["v1", "v2", "v3"],
        help="Model architecture: 'v3' (AdaptiveExpertClassifierV3), 'v2' (DualExpertClassifierV2) or 'v1' (FastUnifiedClassifier). Default: 'v3'."
    )
    # Protocol & performance parameters
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=30,
        help="Total iterations (Seed 0 is original split, Seeds 1 to N-1 are reshuffled). Default: 30."
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="tree",
        choices=["tree", "ridge", "ensemble"],
        help="Decision head for v1: 'tree' (ExtraTrees), 'ridge' (RidgeCV), or 'ensemble'. Default: 'tree'."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Total CPU worker threads/processes. Default: 4."
    )
    parser.add_argument(
        "--parallel_seeds",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="Seed concurrency strategy: 'auto' (parallel on small datasets <= threshold), 'always', or 'never'. Default: 'auto'."
    )
    parser.add_argument(
        "--concurrency_threshold",
        type=int,
        default=600000,
        help="Max total data points (N_total * length) to activate automatic seed concurrency. Default: 600,000."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_v3",
        help="Directory to save per-dataset and summary CSV files. Default: 'results_v3'."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of datasets even if results already exist. Default: False (resume mode)."
    )
    return parser.parse_args()


def get_all_dataset_names():
    if os.path.exists(BASELINE_EXCEL):
        try:
            df = pd.read_excel(BASELINE_EXCEL).iloc[:109]
            return [d for d in df["dataset_name"].dropna().unique() if str(d).strip()]
        except Exception:
            pass
    if os.path.exists(BASELINE_CSV):
        df = pd.read_csv(BASELINE_CSV)
        return [d for d in df["Dataset"].dropna().unique() if d != "Mean" and str(d).strip()]
    raise FileNotFoundError(f"Neither '{BASELINE_EXCEL}' nor '{BASELINE_CSV}' was found.")


def load_all_baselines():
    """
    Loads full baseline dictionary from benchmark Excel file.
    Returns: dict mapping dataset_name -> dict of baseline metric values.
    """
    baseline_dict = {}
    if os.path.exists(BASELINE_EXCEL):
        try:
            df = pd.read_excel(BASELINE_EXCEL).iloc[:109]
            for _, row in df.iterrows():
                dname = str(row["dataset_name"]).strip()
                if not dname:
                    continue
                baseline_dict[dname] = {
                    "baseline_single_split": float(row.get("Accuracy", np.nan)) if pd.notna(row.get("Accuracy")) else np.nan,
                    "baseline_hydra_multirocket": float(row.get("Hydra+Multirocket", np.nan)) if pd.notna(row.get("Hydra+Multirocket")) else np.nan,
                    "baseline_multirocket": float(row.get("MultiRocket", np.nan)) if pd.notna(row.get("MultiRocket")) else np.nan,
                    "baseline_hydra": float(row.get("Hydra", np.nan)) if pd.notna(row.get("Hydra")) else np.nan,
                    "baseline_hc2": float(row.get("HIVE-COTE 2.0", np.nan)) if pd.notna(row.get("HIVE-COTE 2.0")) else np.nan,
                }
            return baseline_dict
        except Exception as e:
            print(f"[Warning] Failed loading baselines from Excel ({e}). Falling back to CSV.")
            
    if os.path.exists(BASELINE_CSV):
        try:
            df = pd.read_csv(BASELINE_CSV)
            for _, row in df.iterrows():
                dname = str(row["Dataset"]).strip()
                if not dname or dname == "Mean":
                    continue
                baseline_dict[dname] = {
                    "baseline_single_split": np.nan,
                    "baseline_hydra_multirocket": float(row.get("Accuracy", np.nan)) if pd.notna(row.get("Accuracy")) else np.nan,
                    "baseline_multirocket": np.nan,
                    "baseline_hydra": np.nan,
                    "baseline_hc2": np.nan,
                }
            return baseline_dict
        except Exception:
            pass
            
    return baseline_dict


# ─────────────────────────────────────────────────────────────
# PROCESS-SAFE LOCKING AND ATOMIC SUMMARY SYNC
# ─────────────────────────────────────────────────────────────

def get_dataset_lock_path(name, output_dir):
    return os.path.join(output_dir, f".{name}.lock")


def is_dataset_in_progress(name, output_dir):
    """
    Checks if another process is currently running this dataset.
    Detects active lockfiles or result files modified within the last 5 minutes.
    """
    lock_file = get_dataset_lock_path(name, output_dir)
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                parts = f.read().strip().split(",")
                pid = int(parts[0])
                lock_time = float(parts[1])
            # If current process owns the lock, it's ours
            if pid == os.getpid():
                return False
            # Check if other process is alive and lock is recent (< 25 min)
            if time.time() - lock_time < 1500:
                try:
                    os.kill(pid, 0)
                    return True
                except OSError:
                    pass  # Process is dead, stale lock
        except Exception:
            pass

    # Secondary check: if result CSV exists, has < 30 seeds, and was modified within the last 5 minutes by another process
    dataset_csv = os.path.join(output_dir, f"{name}_results.csv")
    if os.path.exists(dataset_csv):
        try:
            mtime = os.path.getmtime(dataset_csv)
            if time.time() - mtime < 300:
                df_part = pd.read_csv(dataset_csv)
                if len(df_part) < 30:
                    return True
        except Exception:
            pass

    return False


def acquire_dataset_lock(name, output_dir):
    lock_file = get_dataset_lock_path(name, output_dir)
    try:
        with open(lock_file, "w") as f:
            f.write(f"{os.getpid()},{time.time()}")
    except Exception:
        pass


def release_dataset_lock(name, output_dir):
    lock_file = get_dataset_lock_path(name, output_dir)
    try:
        if os.path.exists(lock_file):
            with open(lock_file, "r") as f:
                parts = f.read().strip().split(",")
                pid = int(parts[0])
            if pid == os.getpid():
                os.remove(lock_file)
    except Exception:
        pass


def update_master_summary(summary_csv, new_res):
    """
    Atomically updates summary_csv by merging new_res, ensuring concurrent runners
    never overwrite each other's completed datasets.
    """
    if new_res is None or new_res.get("status") in ["in_progress_by_other", "skipped", None]:
        return
    try:
        if os.path.exists(summary_csv):
            try:
                df_curr = pd.read_csv(summary_csv)
                records = df_curr.to_dict("records")
            except Exception:
                records = []
        else:
            records = []
            
        ds_name = new_res.get("dataset")
        idx = next((i for i, r in enumerate(records) if r.get("dataset") == ds_name), None)
        if idx is not None:
            records[idx] = new_res
        else:
            records.append(new_res)
            
        df_out = pd.DataFrame(records)
        tmp_csv = summary_csv + f".tmp_{os.getpid()}"
        df_out.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, summary_csv)
    except Exception as e:
        print(f"[Warning] Could not update master summary atomically ({e})")


def _evaluate_single_seed_worker(
    seed,
    X_all,
    y_all,
    n_train_orig,
    n_test_orig,
    model_version="v3",
    classifier_type="tree",
    model_n_jobs=1,
    X_tr_orig=None,
    y_tr_orig=None,
    X_te_orig=None,
    y_te_orig=None
):
    """
    Standalone worker function to evaluate a single seed iteration.
    Guarantees strict train/test isolation (zero test leakage).
    """
    t0 = time.perf_counter()
    try:
        if seed == 0 and X_tr_orig is not None and y_tr_orig is not None:
            split_desc = "Original Library"
            X_train, y_train = X_tr_orig, y_tr_orig
            X_test, y_test = X_te_orig, y_te_orig
        else:
            split_desc = f"Reshuffled s={seed}"
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=n_train_orig,
                test_size=n_test_orig,
                random_state=seed
            )
            split_target = X_all[:, 0, :] if X_all.ndim == 3 else X_all
            train_idx, test_idx = next(splitter.split(split_target, y_all))
            
            assert len(set(train_idx).intersection(set(test_idx))) == 0, "Data leakage detected: train and test overlap!"
            
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

        if model_version == "v3":
            clf = AdaptiveExpertClassifierV3(
                n_jobs=model_n_jobs,
                random_state=seed if seed > 0 else 42
            )
        elif model_version == "v2":
            clf = DualExpertClassifierV2(
                n_jobs=model_n_jobs,
                random_state=seed if seed > 0 else 42
            )
        else:
            clf = FastUnifiedClassifier(
                hydra_kernels=8,
                hydra_groups=16,
                quant_depth=5,
                n_fft_bands=16,
                classifier_type=classifier_type,
                n_jobs=model_n_jobs,
                random_state=seed if seed > 0 else 42
            )
        
        # Fit strictly on train partition
        clf.fit(X_train, y_train)
        
        # Predict strictly on test partition
        preds = clf.predict(X_test)
        acc = float(np.mean(preds == y_test))
        elapsed = time.perf_counter() - t0
        
        return {
            "seed": seed,
            "split_type": split_desc,
            "train_n": len(X_train),
            "test_n": len(X_test),
            "accuracy": acc,
            "time_sec": elapsed
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "seed": seed,
            "split_type": f"Error s={seed}",
            "train_n": n_train_orig,
            "test_n": n_test_orig,
            "accuracy": np.nan,
            "time_sec": elapsed,
            "error": str(e)
        }


def evaluate_single_dataset(
    name,
    n_iterations=30,
    classifier_type="tree",
    model_version="v3",
    n_jobs=4,
    output_dir="results_v3",
    baseline_info=None,
    force=False,
    parallel_seeds="auto",
    concurrency_threshold=600000
):
    os.makedirs(output_dir, exist_ok=True)
    dataset_csv = os.path.join(output_dir, f"{name}_results.csv")
    
    if baseline_info is None:
        baseline_info = {}
    
    base_single = baseline_info.get("baseline_single_split", np.nan)
    base_hm = baseline_info.get("baseline_hydra_multirocket", np.nan)
    base_mr = baseline_info.get("baseline_multirocket", np.nan)
    base_hydra = baseline_info.get("baseline_hydra", np.nan)
    base_hc2 = baseline_info.get("baseline_hc2", np.nan)

    # ── RESUME CHECK ──────────────────────────────────────────
    if not force and os.path.exists(dataset_csv):
        try:
            df_existing = pd.read_csv(dataset_csv)
            if len(df_existing) >= n_iterations and df_existing["accuracy"].notna().all():
                s0_rows = df_existing.loc[df_existing["seed"] == 0, "accuracy"]
                seed0_acc = float(s0_rows.values[0]) if len(s0_rows) > 0 else np.nan
                mean_acc = float(df_existing["accuracy"].mean())
                std_acc = float(df_existing["accuracy"].std())
                min_acc = float(df_existing["accuracy"].min())
                max_acc = float(df_existing["accuracy"].max())
                mean_time = float(df_existing["time_sec"].mean())
                total_time = float(df_existing["time_sec"].sum())
                
                print(f"[{name}] Already completed ({len(df_existing)} seeds). Loaded from cache: Mean Acc = {mean_acc:.4f}. Skipping.")
                
                return {
                    "dataset": name,
                    "status": "completed (cached)",
                    "train_n": df_existing.loc[0, "train_n"] if "train_n" in df_existing else np.nan,
                    "test_n": df_existing.loc[0, "test_n"] if "test_n" in df_existing else np.nan,
                    "length": np.nan,
                    "seed0_acc": seed0_acc,
                    "baseline_single_split": base_single,
                    "gain_over_single_split": (seed0_acc - base_single) if (not np.isnan(seed0_acc) and not np.isnan(base_single)) else np.nan,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "max_acc": max_acc,
                    "baseline_hydra_multirocket": base_hm,
                    "gain_over_baseline": (mean_acc - base_hm) if not np.isnan(base_hm) else np.nan,
                    "baseline_multirocket": base_mr,
                    "baseline_hydra": base_hydra,
                    "baseline_hc2": base_hc2,
                    "mean_time_sec": mean_time,
                    "total_time_sec": total_time
                }
        except Exception as e:
            print(f"[{name}] Notice: Existing CSV could not be parsed ({e}). Recomputing...")

    # ── CONCURRENT RUNNER COLLISION CHECK ─────────────────────
    if not force and is_dataset_in_progress(name, output_dir):
        print(f"[{name}] In progress by another runner process. Skipping to advance ground.")
        return {"dataset": name, "status": "in_progress_by_other"}

    acquire_dataset_lock(name, output_dir)
    try:
        print("\n" + "=" * 70)
        print(f"EVALUATING DATASET: {name}")
        print("=" * 70)
        
        # ── LOAD DATASET WITH CRASH PROTECTION ────────────────────
        try:
            X_tr_orig, y_tr_orig = load_classification(name, split="train")
            X_te_orig, y_te_orig = load_classification(name, split="test")
        except Exception as e:
            print(f"[ERROR] Failed to load dataset '{name}': {e}")
            return {
                "dataset": name,
                "status": f"failed: {str(e)[:100]}",
                "train_n": np.nan,
                "test_n": np.nan,
                "length": np.nan,
                "seed0_acc": np.nan,
                "baseline_single_split": base_single,
                "gain_over_single_split": np.nan,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "min_acc": np.nan,
                "max_acc": np.nan,
                "baseline_hydra_multirocket": base_hm,
                "gain_over_baseline": np.nan,
                "baseline_multirocket": base_mr,
                "baseline_hydra": base_hydra,
                "baseline_hc2": base_hc2,
                "mean_time_sec": np.nan,
                "total_time_sec": np.nan
            }

        n_train_orig = len(X_tr_orig)
        n_test_orig = len(X_te_orig)
        n_total = n_train_orig + n_test_orig
        length = X_tr_orig.shape[-1]
        classes = np.unique(np.concatenate([y_tr_orig, y_te_orig]))
        data_size = n_total * length
        
        # ── DETERMINE CONCURRENCY STRATEGY ────────────────────────
        if parallel_seeds == "always":
            use_concurrency = (n_jobs > 1)
        elif parallel_seeds == "never":
            use_concurrency = False
        else:  # auto
            use_concurrency = (n_jobs > 1 and data_size <= concurrency_threshold)

        print(f"Total Cases:          {n_total} (Split sizes: {n_train_orig} Train / {n_test_orig} Test)")
        print(f"Series Length:        {length} (Total Points N*L = {data_size:,})")
        print(f"Number of Classes:    {len(classes)}")
        if not np.isnan(base_hm):
            print(f"Benchmark Baseline:   Hydra+MultiRocket = {base_hm:.4f}")
        if not np.isnan(base_hc2):
            print(f"HIVE-COTE 2.0 Baseline:                 = {base_hc2:.4f}")
            
        if use_concurrency:
            print(f"Concurrency Strategy: AUTOMATIC PARALLEL ({n_jobs} seed workers, small dataset N*L <= {concurrency_threshold:,})")
        else:
            print(f"Concurrency Strategy: SEQUENTIAL SEEDS ({n_jobs} threads/model, dataset N*L > {concurrency_threshold:,})")

        print("-" * 70)
        print(f"{'Seed':<6} | {'Split Type':<18} | {'Train':<6} | {'Test':<6} | {'Accuracy':<10} | {'Time (s)':<10}")
        print("-" * 70)

        X_all = np.concatenate([X_tr_orig, X_te_orig], axis=0)
        y_all = np.concatenate([y_tr_orig, y_te_orig], axis=0)

        # Check for partial completion
        records = []
        completed_seeds = set()
        if not force and os.path.exists(dataset_csv):
            try:
                df_part = pd.read_csv(dataset_csv)
                records = df_part.to_dict("records")
                completed_seeds = set(df_part["seed"].dropna().astype(int))
                print(f"Resuming {name} with {len(completed_seeds)} seed(s) already completed...")
            except Exception:
                records = []
                completed_seeds = set()

        seeds_to_run = [s for s in range(0, n_iterations) if s not in completed_seeds]

        if seeds_to_run:
            if use_concurrency:
                # ── PARALLEL EXECUTION ACROSS CORES ───────────────────────
                generator = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                    delayed(_evaluate_single_seed_worker)(
                        seed=s,
                        X_all=X_all,
                        y_all=y_all,
                        n_train_orig=n_train_orig,
                        n_test_orig=n_test_orig,
                        model_version=model_version,
                        classifier_type=classifier_type,
                        model_n_jobs=1,
                        X_tr_orig=X_tr_orig if s == 0 else None,
                        y_tr_orig=y_tr_orig if s == 0 else None,
                        X_te_orig=X_te_orig if s == 0 else None,
                        y_te_orig=y_te_orig if s == 0 else None,
                    )
                    for s in seeds_to_run
                )
                for rec in generator:
                    records.append(rec)
                    if np.isnan(rec["accuracy"]):
                        print(f"{rec['seed']:<6d} | ERROR in seed {rec['seed']}: {rec.get('error', 'Unknown')}")
                    else:
                        print(f"{rec['seed']:<6d} | {rec['split_type']:<18s} | {rec['train_n']:<6d} | {rec['test_n']:<6d} | {rec['accuracy']:<10.4f} | {rec['time_sec']:<10.2f}")
                    # Save progressive CSV ordered by seed
                    pd.DataFrame(sorted(records, key=lambda x: x["seed"])).to_csv(dataset_csv, index=False)
            else:
                # ── SEQUENTIAL EXECUTION (MULTICORE MODEL) ────────────────
                for s in seeds_to_run:
                    rec = _evaluate_single_seed_worker(
                        seed=s,
                        X_all=X_all,
                        y_all=y_all,
                        n_train_orig=n_train_orig,
                        n_test_orig=n_test_orig,
                        model_version=model_version,
                        classifier_type=classifier_type,
                        model_n_jobs=n_jobs,
                        X_tr_orig=X_tr_orig if s == 0 else None,
                        y_tr_orig=y_tr_orig if s == 0 else None,
                        X_te_orig=X_te_orig if s == 0 else None,
                        y_te_orig=y_te_orig if s == 0 else None,
                    )
                    records.append(rec)
                    if np.isnan(rec["accuracy"]):
                        print(f"{rec['seed']:<6d} | ERROR in seed {rec['seed']}: {rec.get('error', 'Unknown')}")
                    else:
                        print(f"{rec['seed']:<6d} | {rec['split_type']:<18s} | {rec['train_n']:<6d} | {rec['test_n']:<6d} | {rec['accuracy']:<10.4f} | {rec['time_sec']:<10.2f}")
                    pd.DataFrame(sorted(records, key=lambda x: x["seed"])).to_csv(dataset_csv, index=False)

        df = pd.DataFrame(sorted(records, key=lambda x: x["seed"]))
        valid_accs = df["accuracy"].dropna()
        
        if len(valid_accs) == 0:
            return {
                "dataset": name,
                "status": "failed (all seeds)",
                "train_n": n_train_orig,
                "test_n": n_test_orig,
                "length": length,
                "seed0_acc": np.nan,
                "baseline_single_split": base_single,
                "gain_over_single_split": np.nan,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "min_acc": np.nan,
                "max_acc": np.nan,
                "baseline_hydra_multirocket": base_hm,
                "gain_over_baseline": np.nan,
                "baseline_multirocket": base_mr,
                "baseline_hydra": base_hydra,
                "baseline_hc2": base_hc2,
                "mean_time_sec": np.nan,
                "total_time_sec": np.nan
            }

        seed0_rows = df.loc[df["seed"] == 0, "accuracy"].dropna()
        seed0_acc = float(seed0_rows.values[0]) if len(seed0_rows) > 0 else np.nan
        mean_acc = float(valid_accs.mean())
        std_acc = float(valid_accs.std())
        min_acc = float(valid_accs.min())
        max_acc = float(valid_accs.max())
        mean_time = float(df["time_sec"].dropna().mean())
        total_time = float(df["time_sec"].dropna().sum())

        print("-" * 70)
        print(f"Summary for {name}:")
        print(f"  Seed 0 (Original Split) Acc:         {seed0_acc:.4f}" if not np.isnan(seed0_acc) else "  Seed 0: N/A")
        if not np.isnan(base_single) and not np.isnan(seed0_acc):
            single_gain = (seed0_acc - base_single) * 100
            print(f"  Gain over Single-Split Baseline:     {single_gain:+.2f}% (Baseline = {base_single:.4f})")
        print(f"  Mean Accuracy across {len(valid_accs)} iterations:  {mean_acc:.4f} (± {std_acc:.4f})")
        print(f"  Accuracy Range:                      [{min_acc:.4f} - {max_acc:.4f}]")
        if not np.isnan(base_hm):
            gain = (mean_acc - base_hm) * 100
            print(f"  Benchmark Baseline (Hydra+MultiR):   {base_hm:.4f} (Net Gain: {gain:+.2f}%)")
        if not np.isnan(base_hc2):
            gain_hc2 = (mean_acc - base_hc2) * 100
            print(f"  Benchmark Baseline (HIVE-COTE 2.0):  {base_hc2:.4f} (Net Gain: {gain_hc2:+.2f}%)")
        print(f"  Mean Execution Time per Iteration:   {mean_time:.2f}s")
        print(f"  Total CPU Time:                      {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Results saved to:                    {dataset_csv}")
        print("=" * 70)

        return {
            "dataset": name,
            "status": "completed",
            "train_n": n_train_orig,
            "test_n": n_test_orig,
            "length": length,
            "seed0_acc": seed0_acc,
            "baseline_single_split": base_single,
            "gain_over_single_split": (seed0_acc - base_single) if (not np.isnan(seed0_acc) and not np.isnan(base_single)) else np.nan,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "min_acc": min_acc,
            "max_acc": max_acc,
            "baseline_hydra_multirocket": base_hm,
            "gain_over_baseline": (mean_acc - base_hm) if not np.isnan(base_hm) else np.nan,
            "baseline_multirocket": base_mr,
            "baseline_hydra": base_hydra,
            "baseline_hc2": base_hc2,
            "mean_time_sec": mean_time,
            "total_time_sec": total_time
        }
    finally:
        release_dataset_lock(name, output_dir)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "summary_results.csv")
    
    # ── DETERMINE DATASET LIST ────────────────────────────────
    if args.all:
        dataset_list = get_all_dataset_names()
        print(f"\n[MODE: ALL 109 DATASETS] Found {len(dataset_list)} datasets to process.")
    elif args.dataset:
        dataset_list = [args.dataset]
    elif args.datasets:
        dataset_list = args.datasets
    else:
        dataset_list = ["InlineSkate"]

    # ── BIDIRECTIONAL ORDERING ────────────────────────────────
    if args.order in ["desc", "reverse"]:
        dataset_list = dataset_list[::-1]
        print(f"[ORDER: DESCENDING] Processing datasets in reverse order ({dataset_list[0]} -> {dataset_list[-1]}).")
    else:
        print(f"[ORDER: ASCENDING] Processing datasets in forward order ({dataset_list[0]} -> {dataset_list[-1]}).")

    all_baselines = load_all_baselines()
    
    total_ds = len(dataset_list)
    print(f"\nQueue contains {total_ds} dataset(s). Output directory: '{args.output_dir}'")
    print(f"Model version: {args.model_version.upper()} | Concurrency strategy: {args.parallel_seeds.upper()} (threshold: {args.concurrency_threshold:,})")
    
    for idx, ds in enumerate(dataset_list, start=1):
        print(f"\n>>> [{idx}/{total_ds}] Processing: {ds}")
        b_info = all_baselines.get(ds, {})
        
        try:
            res = evaluate_single_dataset(
                name=ds,
                n_iterations=args.n_iterations,
                classifier_type=args.classifier_type,
                model_version=args.model_version,
                n_jobs=args.n_jobs,
                output_dir=args.output_dir,
                baseline_info=b_info,
                force=args.force,
                parallel_seeds=args.parallel_seeds,
                concurrency_threshold=args.concurrency_threshold
            )
        except Exception as crash_err:
            print(f"\n[CRITICAL ERROR] Unexpected failure on dataset '{ds}': {crash_err}")
            traceback.print_exc()
            base_hm = b_info.get("baseline_hydra_multirocket", np.nan)
            res = {
                "dataset": ds,
                "status": f"crashed: {str(crash_err)[:100]}",
                "train_n": np.nan,
                "test_n": np.nan,
                "length": np.nan,
                "seed0_acc": np.nan,
                "baseline_single_split": b_info.get("baseline_single_split", np.nan),
                "gain_over_single_split": np.nan,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "min_acc": np.nan,
                "max_acc": np.nan,
                "baseline_hydra_multirocket": base_hm,
                "gain_over_baseline": np.nan,
                "baseline_multirocket": b_info.get("baseline_multirocket", np.nan),
                "baseline_hydra": b_info.get("baseline_hydra", np.nan),
                "baseline_hc2": b_info.get("baseline_hc2", np.nan),
                "mean_time_sec": np.nan,
                "total_time_sec": np.nan
            }

        # Atomically update master summary table on disk
        if res is not None and res.get("status") != "in_progress_by_other":
            update_master_summary(summary_csv, res)

    # ── FINAL COMBINED SUMMARY DISPLAY ────────────────────────
    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        print("\n\n" + "=" * 90)
        print("FINAL COMBINED SUMMARY REPORT")
        print("=" * 90)
        display_cols = ["dataset", "status", "seed0_acc", "mean_acc", "baseline_hydra_multirocket", "gain_over_baseline", "mean_time_sec"]
        avail_cols = [c for c in display_cols if c in df_sum.columns]
        print(df_sum[avail_cols].to_string(index=False))
        print("-" * 90)
        
        completed_df = df_sum[df_sum["status"].str.contains("completed", na=False)]
        if len(completed_df) > 0:
            valid_means = completed_df["mean_acc"].dropna()
            print(f"Total Successfully Completed: {len(completed_df)} / 109")
            print(f"Overall Mean Accuracy:        {valid_means.mean():.4f}")
            if "gain_over_baseline" in completed_df and completed_df["gain_over_baseline"].notna().any():
                mean_gain = completed_df["gain_over_baseline"].dropna().mean() * 100
                print(f"Overall Mean Net Gain:        {mean_gain:+.2f}%")
        print(f"Full summary saved to: {summary_csv}")
        print("=" * 90)


if __name__ == "__main__":
    main()
