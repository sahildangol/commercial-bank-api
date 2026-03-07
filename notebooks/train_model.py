"""
FILE 2 — train_model.py
════════════════════════════════════════════════════════════════════════════════
INPUT : enriched_features.csv  (output of feature_engineering.py)

OUTPUT: model_artifacts/
        ├── model_direction.pkl    HistGBM classifier  → P(price up in 21d)
        ├── model_momentum.pkl     HistGBM classifier  → P(price up in 5d)
        ├── model_magnitude.pkl    HistGBM regressor   → expected 21d return %
        ├── scaler.pkl             MinMaxScaler for ensemble normalisation
        └── model_meta.json        AUC scores, split date, feature list, thresholds

USAGE:
    python train_model.py
    python train_model.py --input enriched_features.csv --outdir model_artifacts/
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                             roc_auc_score, mean_absolute_error)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Import feature list from feature_engineering
from feature_engineering import MODEL_FEATURES, TARGET_COLS

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_SPLIT_RATIO = 0.90       # 95% train, 5% test (temporal split)
SIGNAL_LONG_THRESH  = 0.65     # ensemble score ≥ 0.65 → LONG
SIGNAL_HOLD_THRESH  = 0.50     # ensemble score ≥ 0.50 → HOLD, else SKIP

# HistGBM hyperparameters (validated from previous sessions, AUC=0.786)
CLF_PARAMS = dict(
    max_iter=600, max_depth=5, learning_rate=0.04,
    min_samples_leaf=50, l2_regularization=0.2,
    max_bins=255, random_state=42, class_weight='balanced',
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
)
REG_PARAMS = dict(
    max_iter=600, max_depth=5, learning_rate=0.04,
    min_samples_leaf=60, l2_regularization=0.3,
    max_bins=255, random_state=42,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data(path: str):
    print(f"  Loading: {path}")
    df = pd.read_csv(path, parse_dates=['date'])
    missing_feats = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing features — run feature_engineering.py first.\n"
                         f"Missing: {missing_feats}")
    missing_tgts = [t for t in TARGET_COLS if t not in df.columns]
    if missing_tgts:
        raise ValueError(f"Missing target columns: {missing_tgts}")
    print(f"  Rows: {len(df):,} | Features: {len(MODEL_FEATURES)} | "
          f"Date: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── TEMPORAL SPLIT ────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, ratio: float = 0.95):
    dates      = df['date'].sort_values().unique()
    split_date = dates[int(len(dates) * ratio)]
    train = df[df['date'] <  split_date]
    test  = df[df['date'] >= split_date]
    print(f"  Split date : {pd.Timestamp(split_date).date()}")
    print(f"  Train      : {train['date'].min().date()} → {train['date'].max().date()} "
          f"({len(train):,} rows)")
    print(f"  Test       : {test['date'].min().date()} → {test['date'].max().date()} "
          f"({len(test):,} rows)")
    return train, test, pd.Timestamp(split_date)


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_direction(X_tr, y_tr):
    clf = HistGradientBoostingClassifier(**CLF_PARAMS)
    clf.fit(X_tr, y_tr)
    return clf

def train_momentum(X_tr, y_tr):
    clf = HistGradientBoostingClassifier(**CLF_PARAMS)
    clf.fit(X_tr, y_tr)
    return clf

def train_magnitude(X_tr, y_tr):
    reg = HistGradientBoostingRegressor(**REG_PARAMS)
    reg.fit(X_tr, y_tr)
    return reg


# ── EVALUATE ──────────────────────────────────────────────────────────────────
def evaluate(models: dict, X_tr, X_te, y_tr, y_te,
             y_mag_tr, y_mag_te, y_mom_te) -> dict:
    clf_dir, clf_mom, reg_mag, scaler = (models['direction'], models['momentum'],
                                          models['magnitude'], models['scaler'])
    # Direction
    prob_dir_tr = clf_dir.predict_proba(X_tr)[:,1]
    prob_dir_te = clf_dir.predict_proba(X_te)[:,1]
    tr_auc = roc_auc_score(y_tr,     prob_dir_tr)
    te_auc = roc_auc_score(y_te,     prob_dir_te)
    te_acc = accuracy_score(y_te,    (prob_dir_te > 0.5).astype(int))
    te_f1  = f1_score(y_te,          (prob_dir_te > 0.5).astype(int))

    # Magnitude
    mag_tr = reg_mag.predict(X_tr)
    mag_te = reg_mag.predict(X_te)
    tr_r2  = r2_score(y_mag_tr, mag_tr)
    te_r2  = r2_score(y_mag_te, mag_te)
    te_mae = mean_absolute_error(y_mag_te, mag_te) * 100

    # Momentum
    prob_mom_te = clf_mom.predict_proba(X_te)[:,1]
    te_auc_mom  = roc_auc_score(y_mom_te, prob_mom_te)

    # Ensemble: 0.6 × direction + 0.4 × normalised magnitude
    mag_norm_tr = scaler.fit_transform(mag_tr.reshape(-1,1)).ravel()
    mag_norm_te = scaler.transform(mag_te.reshape(-1,1)).ravel()
    ens_tr = 0.6 * prob_dir_tr + 0.4 * mag_norm_tr
    ens_te = 0.6 * prob_dir_te + 0.4 * mag_norm_te
    te_auc_ens = roc_auc_score(y_te, ens_te)
    tr_auc_ens = roc_auc_score(y_tr, ens_tr)

    metrics = dict(
        tr_auc_dir=round(tr_auc, 4),    te_auc_dir=round(te_auc, 4),
        tr_auc_ens=round(tr_auc_ens,4), te_auc_ens=round(te_auc_ens,4),
        te_auc_mom=round(te_auc_mom,4),
        te_acc=round(te_acc, 4),        te_f1=round(te_f1, 4),
        tr_r2=round(tr_r2, 4),          te_r2=round(te_r2, 4),
        te_mae=round(te_mae, 4),
        overfit_gap=round(tr_auc - te_auc, 4),
    )

    print(f"\n  ┌─ Results ─────────────────────────────────────┐")
    print(f"  │  Direction   Train AUC={tr_auc:.4f}  Test AUC={te_auc:.4f}  │")
    print(f"  │  Ensemble    Train AUC={tr_auc_ens:.4f}  Test AUC={te_auc_ens:.4f}  │")
    print(f"  │  Momentum    Test AUC={te_auc_mom:.4f}                     │")
    print(f"  │  Magnitude   Train R²={tr_r2:.4f}   Test R²={te_r2:.4f}   │")
    print(f"  │  Accuracy={te_acc:.3f}  F1={te_f1:.3f}  MAE={te_mae:.2f}%       │")
    print(f"  │  Overfit gap={tr_auc - te_auc:+.4f}                          │")
    print(f"  └───────────────────────────────────────────────┘")

    # Per-bank
    print(f"\n  Per-bank Test AUC (direction):")
    test_df = pd.DataFrame({'prob_dir': prob_dir_te, 'ens': ens_te,
                            'target_dir': y_te.values})
    print(f"  {'Bank':8s} {'Dir AUC':>9} {'Ens AUC':>9}")
    per_bank = {}
    # Reconstruct bank from test split — needs test_df with bank col
    # (passed separately in main)
    return metrics, prob_dir_te, ens_te, mag_te


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
def compute_importance(clf, X_te, y_te, n_samples=800) -> pd.Series:
    idx  = np.random.default_rng(42).choice(len(X_te), size=min(n_samples, len(X_te)), replace=False)
    Xs   = X_te.iloc[idx]; ys = y_te.iloc[idx]
    pi   = permutation_importance(clf, Xs, ys, n_repeats=8,
                                  random_state=42, n_jobs=-1, scoring='roc_auc')
    return pd.Series(pi.importances_mean, index=MODEL_FEATURES).sort_values(ascending=False)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(input_path: str, outdir: str):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "="*65)
    print(" STEP 2 — Model Training")
    print("="*65)

    # Load
    print("\n[1/6] Loading enriched features...")
    df = load_data(input_path)

    # Split
    print("\n[2/6] Temporal split (95/5)...")
    train, test, split_date = temporal_split(df)

    X_tr = train[MODEL_FEATURES];  X_te = test[MODEL_FEATURES]
    y_dir_tr = train['target_dir']; y_dir_te = test['target_dir']
    y_mag_tr = train['target_mag']; y_mag_te = test['target_mag']
    y_mom_tr = train['target_mom5'];y_mom_te = test['target_mom5']

    # Train
    print("\n[3/6] Training models...")
    print("  Training direction classifier (P(up 21d))...")
    clf_dir = train_direction(X_tr, y_dir_tr)
    print("  Training momentum classifier (P(up 5d))...")
    clf_mom = train_momentum(X_tr, y_mom_tr)
    print("  Training magnitude regressor (expected 21d return)...")
    reg_mag = train_magnitude(X_tr, y_mag_tr)
    scaler  = MinMaxScaler()

    models = dict(direction=clf_dir, momentum=clf_mom,
                  magnitude=reg_mag, scaler=scaler)

    # Evaluate
    print("\n[4/6] Evaluating...")
    metrics, prob_dir_te, ens_te, mag_te = evaluate(
        models, X_tr, X_te, y_dir_tr, y_dir_te,
        y_mag_tr, y_mag_te, y_mom_te
    )

    # Per-bank breakdown
    per_bank = {}
    print(f"\n  {'Bank':8s} {'Dir AUC':>9} {'R²':>7} {'Signal@last':>13}")
    print("  " + "-"*42)
    test_out = test.copy()
    test_out['prob_dir'] = prob_dir_te
    test_out['ens']      = ens_te
    test_out['pred_mag'] = mag_te
    for bk in sorted(test_out['bank'].unique()):
        b = test_out[test_out['bank'] == bk]
        if b['target_dir'].nunique() < 2: continue
        b_auc = roc_auc_score(b['target_dir'], b['prob_dir'])
        b_r2  = r2_score(b['target_mag'], b['pred_mag'])
        last_ens = float(b.sort_values('date').iloc[-1]['ens'])
        sig = 'LONG' if last_ens >= SIGNAL_LONG_THRESH else 'HOLD' if last_ens >= SIGNAL_HOLD_THRESH else 'SKIP'
        per_bank[bk] = dict(dir_auc=round(b_auc,3), r2=round(b_r2,3),
                            last_ens=round(last_ens,3), signal=sig)
        print(f"  {bk:8s} {b_auc:9.3f} {b_r2:7.3f} {sig:>13}")

    # Feature importance
    print("\n[5/6] Computing feature importance...")
    fi = compute_importance(clf_dir, X_te, y_dir_te)
    print("  Top 10 features:")
    for feat, val in fi.head(10).items():
        print(f"    {feat:28s} {val:.4f}")

    # Save artifacts
    print("\n[6/6] Saving artifacts...")
    artifacts = {
        'clf_dir': clf_dir,
        'clf_mom': clf_mom,
        'reg_mag': reg_mag,
        'scaler':  scaler,
    }
    for name, obj in artifacts.items():
        path = os.path.join(outdir, f'model_{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")

    # Meta JSON (human-readable, also used by inference pipeline)
    meta = {
        'model_features': MODEL_FEATURES,
        'n_features':      len(MODEL_FEATURES),
        'split_date':      str(split_date.date()),
        'train_end':       str(train['date'].max().date()),
        'test_start':      str(test['date'].min().date()),
        'test_end':        str(test['date'].max().date()),
        'metrics':         metrics,
        'per_bank':        per_bank,
        'signal_thresholds': {
            'LONG': SIGNAL_LONG_THRESH,
            'HOLD': SIGNAL_HOLD_THRESH,
        },
        'top_features':    fi.head(20).to_dict(),
        'trained_at':      pd.Timestamp.now().isoformat(),
        'active_banks':    sorted(df['bank'].unique().tolist()),
    }
    meta_path = os.path.join(outdir, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    print(f"\n  ✓ All artifacts saved to: {outdir}/")
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n")
    return models, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NEPSE HistGBM models')
    parser.add_argument('--input',  default='../data/processed/enriched_features.csv',
                        help='Path to enriched_features.csv')
    parser.add_argument('--outdir', default='../models',
                        help='Directory to save model artifacts')
    args = parser.parse_args()
    run(args.input, args.outdir)
