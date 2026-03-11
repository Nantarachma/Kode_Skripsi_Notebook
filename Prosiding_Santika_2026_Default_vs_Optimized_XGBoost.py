# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Prosiding SANTIKA 2026 — Pengaruh Optimasi Hiperparameter Bayesian TPE     ║
# ║  terhadap Kinerja Klasifikasi XGBoost pada Dataset NF-UNSW-NB15-v3         ║
# ║                                                                              ║
# ║  Perbandingan: Default XGBoost vs TPE-Optimized XGBoost                    ║
# ║  Optimasi: SINGLE OBJECTIVE — Maximize Macro F1-Score                      ║
# ║  Metode: Bayesian TPE (Tree-structured Parzen Estimator) via Optuna        ║
# ║                                                                              ║
# ║  Catatan: Script ini merupakan kode lengkap yang dapat dijalankan          ║
# ║  end-to-end. Sesuaikan variabel DATA_PATH dengan lokasi file CSV.          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ═══ CELL 1 ═══ Import & GPU Setup ═══════════════════════════════════════════

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, cohen_kappa_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestRegressor

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Konstanta Global ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_TRIALS    = 30
TARGET_COL  = 'mapped_label'
TARGET_NAMES = ['Normal', 'DoS', 'Probe', 'Malware']
DATA_PATH   = 'NF-UNSW-NB15-v3.csv'   # ganti dengan path lengkap ke file, contoh:
                                       # '/kaggle/input/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv'

# ── Reprodusibilitas ─────────────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)

# ── Verifikasi GPU ────────────────────────────────────────────────────────────
print("=" * 60)
print("CELL 1: Import & GPU Setup")
print("=" * 60)
try:
    _test_model = xgb.XGBClassifier(
        tree_method='hist', device='cuda',
        n_estimators=1, random_state=RANDOM_SEED
    )
    _X_tmp = np.random.rand(100, 10).astype('float32')
    _y_tmp = np.random.randint(0, 2, 100)
    _test_model.fit(_X_tmp, _y_tmp, verbose=False)
    print("✅ GPU XGBoost tersedia — device='cuda', tree_method='hist'")
    DEVICE = 'cuda'
except Exception:
    print("⚠️  GPU tidak tersedia — fallback ke CPU")
    DEVICE = 'cpu'

print(f"Random Seed : {RANDOM_SEED}")
print(f"N Trials TPE: {N_TRIALS}")
print(f"XGBoost ver : {xgb.__version__}")
print(f"Optuna ver  : {optuna.__version__}")


# ═══ CELL 2 ═══ Load Dataset NF-UNSW-NB15-v3 + Mapping 10 → 4 Kelas ════════

print("\n" + "=" * 60)
print("CELL 2: Load Dataset & Mapping Kelas")
print("=" * 60)

t0 = time.time()
df_full = pd.read_csv(DATA_PATH)
load_time = time.time() - t0
print(f"Dataset dimuat dalam {load_time:.2f} detik")
print(f"Dimensi awal : {df_full.shape[0]:,} baris × {df_full.shape[1]} kolom")

# ── Mapping 10 kelas asli → 4 kelas ──────────────────────────────────────────
mapping_rules = {
    'Benign'       : 0,                # Normal
    'DoS'          : 1, 'Generic'  : 1, # DoS
    'Reconnaissance': 2, 'Analysis': 2, # Probe
    'Exploits'     : 3, 'Fuzzers'  : 3,
    'Backdoor'     : 3, 'Shellcode': 3,
    'Worms'        : 3                  # Malware
}

df_full['mapped_label'] = df_full['Attack'].map(mapping_rules).astype(int)

print("\nDistribusi kelas setelah mapping:")
dist = df_full['mapped_label'].value_counts().sort_index()
for idx, count in dist.items():
    pct = count / len(df_full) * 100
    print(f"  Kelas {idx} ({TARGET_NAMES[idx]:8s}): {count:>10,} sampel ({pct:.2f}%)")

# ── Split awal 80:20 (train+val : test holdout) ───────────────────────────────
df_trainval, df_test_raw = train_test_split(
    df_full, test_size=0.2, random_state=RANDOM_SEED,
    stratify=df_full['mapped_label']
)
print(f"\nSplit awal — Train+Val: {len(df_trainval):,} | Test: {len(df_test_raw):,}")


# ═══ CELL 3 ═══ Preprocessing ════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CELL 3: Preprocessing")
print("=" * 60)

# ── Kolom yang di-drop ────────────────────────────────────────────────────────
COLS_TO_DROP = [
    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'
]

df_trainval_clean = df_trainval.drop(columns=COLS_TO_DROP, errors='ignore')
df_test_clean     = df_test_raw.drop(columns=COLS_TO_DROP, errors='ignore')

# ── Pisahkan fitur dan target ─────────────────────────────────────────────────
feature_cols = [c for c in df_trainval_clean.columns
                if c not in [TARGET_COL, 'Attack']]

X_raw_trainval = df_trainval_clean[feature_cols].copy()
y_trainval     = df_trainval_clean[TARGET_COL].values

X_raw_test     = df_test_clean[feature_cols].copy()
y_test         = df_test_clean[TARGET_COL].values

# ── Ganti inf dengan NaN, lalu imputasi dengan median (dari data train) ───────
print(f"Jumlah fitur setelah drop : {len(feature_cols)}")

X_raw_trainval.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw_test.replace([np.inf, -np.inf], np.nan, inplace=True)

nan_before = X_raw_trainval.isna().sum().sum()
train_medians = X_raw_trainval.median()
X_raw_trainval.fillna(train_medians, inplace=True)
X_raw_test.fillna(train_medians, inplace=True)
nan_after = X_raw_trainval.isna().sum().sum()

print(f"NaN sebelum imputasi      : {nan_before:,}")
print(f"NaN setelah imputasi      : {nan_after}")
print("✅ Data 100% bersih dari NaN dan Infinity")

# ── StandardScaler — fit pada train, transform semua ─────────────────────────
scaler = StandardScaler()
X_scaled_trainval = pd.DataFrame(
    scaler.fit_transform(X_raw_trainval),
    columns=feature_cols
)
X_scaled_test = pd.DataFrame(
    scaler.transform(X_raw_test),
    columns=feature_cols
)
print("✅ StandardScaler — fit_on_train, transform_all selesai")


# ═══ CELL 4 ═══ Split Train:Val (80:20) & Konversi float32 ══════════════════

print("\n" + "=" * 60)
print("CELL 4: Split Train:Val & Konversi float32")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_trainval, y_trainval,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_trainval
)

X_test_selected = X_scaled_test.copy()
y_test_final    = y_test

# ── Konversi float64 → float32 ────────────────────────────────────────────────
for _df in [X_train, X_val, X_test_selected]:
    float_cols = _df.select_dtypes(include=['float64']).columns
    _df[float_cols] = _df[float_cols].astype('float32')

print(f"Train  : {len(X_train):>10,} sampel | {X_train.shape[1]} fitur")
print(f"Val    : {len(X_val):>10,} sampel | {X_val.shape[1]} fitur")
print(f"Test   : {len(X_test_selected):>10,} sampel | {X_test_selected.shape[1]} fitur")
print("✅ Konversi float32 selesai")


# ═══ CELL 5 ═══ Hybrid Cost-Sensitive Weighting ══════════════════════════════

print("\n" + "=" * 60)
print("CELL 5: Hybrid Cost-Sensitive Weighting")
print("=" * 60)

# Langkah 1: compute_sample_weight 'balanced'
raw_weights          = compute_sample_weight(class_weight='balanced', y=y_train)
# Langkah 2: sqrt untuk meredam penalti ekstrem
sqrt_weights         = np.sqrt(raw_weights)
# Langkah 3: normalisasi agar rata-rata = 1.0
sample_weights_train = sqrt_weights / sqrt_weights.mean()

print("Statistik bobot hybrid per kelas:")
for cls_id, cls_name in enumerate(TARGET_NAMES):
    mask     = y_train == cls_id
    n_samples = mask.sum()
    w_mean   = sample_weights_train[mask].mean()
    print(f"  Kelas {cls_id} ({cls_name:8s}): {n_samples:>10,} sampel | bobot = {w_mean:.4f}")

print(f"\nMin  bobot: {sample_weights_train.min():.4f}")
print(f"Max  bobot: {sample_weights_train.max():.4f}")
print(f"Mean bobot: {sample_weights_train.mean():.4f}")


# ═══ CELL 6 ═══ Default XGBoost Training ════════════════════════════════════

print("\n" + "=" * 60)
print("CELL 6: Default XGBoost Training")
print("=" * 60)

DEFAULT_PARAMS = {
    'n_estimators'     : 100,
    'learning_rate'    : 0.3,
    'max_depth'        : 6,
    'min_child_weight' : 1,
    'max_delta_step'   : 0,
    'gamma'            : 0,
    'subsample'        : 1.0,
    'colsample_bytree' : 1.0,
    'reg_alpha'        : 0,
    'reg_lambda'       : 1,
    'objective'        : 'multi:softmax',
    'num_class'        : len(TARGET_NAMES),
    'tree_method'      : 'hist',
    'device'           : DEVICE,
    'eval_metric'      : 'mlogloss',
    'verbosity'        : 0,
    'random_state'     : RANDOM_SEED,
}

print("Parameter Default XGBoost:")
for k, v in DEFAULT_PARAMS.items():
    if k not in ['objective', 'num_class', 'tree_method', 'device',
                 'eval_metric', 'verbosity', 'random_state']:
        print(f"  {k:20s} = {v}")

t_start_default = time.time()
default_model = xgb.XGBClassifier(**DEFAULT_PARAMS)
default_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
train_time_default = time.time() - t_start_default

print(f"\n✅ Default XGBoost dilatih dalam {train_time_default:.2f} detik")

# ── Validasi F1 Default ───────────────────────────────────────────────────────
val_preds_default  = default_model.predict(X_val)
val_f1_default     = f1_score(y_val, val_preds_default, average='macro')
print(f"Validasi Macro F1 Default: {val_f1_default:.4f}")


# ═══ CELL 7 ═══ TPE Optimization — SINGLE OBJECTIVE ═══════════════════════

print("\n" + "=" * 60)
print("CELL 7: TPE Optimization — Single Objective (Maximize F1)")
print("=" * 60)

# ── Fungsi Objektif: hanya kembalikan f1_macro (SCALAR) ──────────────────────
def objective_tpe_single(trial):
    param = {
        'n_estimators'     : trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 6, 12),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 7),
        'max_delta_step'   : trial.suggest_int('max_delta_step', 1, 8),
        'gamma'            : trial.suggest_float('gamma', 0.1, 0.5),
        'subsample'        : trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
        'objective'        : 'multi:softmax',
        'num_class'        : len(TARGET_NAMES),
        'tree_method'      : 'hist',
        'device'           : DEVICE,
        'eval_metric'      : 'mlogloss',
        'verbosity'        : 0,
        'random_state'     : RANDOM_SEED,
    }

    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds    = model.predict(X_val)
    f1_macro = f1_score(y_val, preds, average='macro')

    # ── SINGLE OBJECTIVE: kembalikan f1_macro saja (scalar) ──────────────────
    return f1_macro


# ── Create Study — SINGLE OBJECTIVE: direction="maximize" ───────────────────
study_tpe = optuna.create_study(
    study_name="TPE_SingleObjective_F1",
    direction="maximize",          # BUKAN directions=["maximize","minimize"]
    sampler=TPESampler(seed=RANDOM_SEED)
)

print(f"Menjalankan {N_TRIALS} trial TPE (single-objective)...")
t_start_tpe = time.time()
study_tpe.optimize(objective_tpe_single, n_trials=N_TRIALS, show_progress_bar=True)
tpe_optim_time = time.time() - t_start_tpe

print(f"\n✅ Optimasi TPE selesai dalam {tpe_optim_time:.2f} detik ({tpe_optim_time/60:.2f} menit)")
print(f"Best trial  : #{study_tpe.best_trial.number}")
print(f"Best Val F1 : {study_tpe.best_trial.value:.4f}")
print("\nParameter terbaik TPE:")
for k, v in study_tpe.best_trial.params.items():
    print(f"  {k:20s} = {v}")


# ═══ CELL 8 ═══ Retrain Model Optimized ═════════════════════════════════════

print("\n" + "=" * 60)
print("CELL 8: Retrain Model Optimized (Best Trial Params)")
print("=" * 60)

# ── Ambil parameter dari study.best_trial (BUKAN study.best_trials) ──────────
best_params = study_tpe.best_trial.params.copy()   # scalar, bukan list
best_params.update({
    'objective'    : 'multi:softmax',
    'num_class'    : len(TARGET_NAMES),
    'tree_method'  : 'hist',
    'device'       : DEVICE,
    'eval_metric'  : 'mlogloss',
    'verbosity'    : 0,
    'random_state' : RANDOM_SEED,
})

t_start_opt = time.time()
optimized_model = xgb.XGBClassifier(**best_params)
optimized_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
train_time_opt = time.time() - t_start_opt
print(f"✅ Model optimized dilatih dalam {train_time_opt:.2f} detik")


# ═══ CELL 9 ═══ Evaluasi Default vs Optimized ═══════════════════════════════

print("\n" + "=" * 60)
print("CELL 9: Evaluasi Default vs Optimized (Test Set)")
print("=" * 60)

# ── Prediksi ──────────────────────────────────────────────────────────────────
preds_default   = default_model.predict(X_test_selected)
preds_optimized = optimized_model.predict(X_test_selected)

# ── Metrik Agregat ────────────────────────────────────────────────────────────
metrics = {}
for label, preds in [('Default', preds_default), ('Optimized (TPE)', preds_optimized)]:
    metrics[label] = {
        'f1_macro'  : f1_score(y_test_final, preds, average='macro'),
        'accuracy'  : accuracy_score(y_test_final, preds),
        'kappa'     : cohen_kappa_score(y_test_final, preds),
    }

print("\n" + "=" * 65)
print(f"{'Metrik':<25} {'Default':>12} {'Optimized (TPE)':>16} {'Δ':>10}")
print("-" * 65)
for metric in ['f1_macro', 'accuracy', 'kappa']:
    d_val = metrics['Default'][metric]
    o_val = metrics['Optimized (TPE)'][metric]
    delta = o_val - d_val
    print(f"  {metric:<23} {d_val:>12.4f} {o_val:>16.4f} {delta:>+10.4f}")

# ── Classification Report ─────────────────────────────────────────────────────
print("\n── Classification Report: Default XGBoost ──")
print(classification_report(
    y_test_final, preds_default,
    target_names=TARGET_NAMES, digits=4
))

print("── Classification Report: TPE-Optimized XGBoost ──")
print(classification_report(
    y_test_final, preds_optimized,
    target_names=TARGET_NAMES, digits=4
))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, preds) in zip(axes, [('Default', preds_default),
                                       ('TPE-Optimized', preds_optimized)]):
    cm = confusion_matrix(y_test_final, preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
    ax.set_title(f'Confusion Matrix (Normalized)\n{label}')
    ax.set_ylabel('Kelas Aktual')
    ax.set_xlabel('Kelas Prediksi')
plt.tight_layout()
plt.savefig('confusion_matrix_default_vs_optimized.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix disimpan: confusion_matrix_default_vs_optimized.png")


# ═══ CELL 11 ═══ HP Importance via Surrogate RF ═════════════════════════════

print("\n" + "=" * 60)
print("CELL 11: HP Importance via Surrogate RF (terhadap F1 saja)")
print("=" * 60)

def get_rf_importance_single(study):
    """
    Hitung importance hiperparameter terhadap F1-Score menggunakan
    Random Forest Regressor sebagai surrogate model.

    Catatan penting: menggunakan t.value (scalar) karena study SINGLE OBJECTIVE.
    BUKAN t.values[objective_index] yang digunakan untuk multi-objective.
    """
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]
    data = []
    for t in trials:
        row = {k: v for k, v in t.params.items()
               if isinstance(v, (int, float))}
        row['target'] = t.value          # t.value — BUKAN t.values[0] / t.values[idx]
        data.append(row)

    df_trials = pd.DataFrame(data)
    X_hp = df_trials.drop(columns='target')
    y_hp = df_trials['target']

    rf_surrogate = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf_surrogate.fit(X_hp, y_hp)
    importances = pd.Series(
        rf_surrogate.feature_importances_,
        index=X_hp.columns
    ).sort_values(ascending=False)
    return importances

hp_importance = get_rf_importance_single(study_tpe)

print("Importance Hiperparameter terhadap Macro F1-Score (TPE):")
for rank, (hp, imp) in enumerate(hp_importance.items(), 1):
    bar = '█' * int(imp * 50)
    print(f"  {rank:2d}. {hp:20s} {imp:.4f}  {bar}")

# ── Visualisasi ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
hp_importance.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Hyperparameter Importance terhadap Macro F1-Score\n(TPE — Single Objective, Surrogate RF)', fontsize=11)
ax.set_xlabel('Importance (Relative)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('hp_importance_tpe_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ HP importance chart disimpan: hp_importance_tpe_f1.png")


# ═══ CELL 11B ═══ Optimization Convergence Analysis ═════════════════════════

print("\n" + "=" * 60)
print("CELL 11B: Optimization Convergence Analysis")
print("=" * 60)

# ── Extract trial data ────────────────────────────────────────────────────────
trial_numbers = []
trial_f1_values = []
best_so_far = []
current_best = -1

for t in study_tpe.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        trial_numbers.append(t.number + 1)  # 1-indexed
        trial_f1_values.append(t.value)
        current_best = max(current_best, t.value)
        best_so_far.append(current_best)

if not trial_f1_values:
    print("⚠️  Tidak ada trial yang berhasil — lewati analisis konvergensi")
else:
    print(f"Trial pertama F1  : {trial_f1_values[0]:.4f}")
    print(f"Trial terbaik F1  : {max(trial_f1_values):.4f} (Trial #{study_tpe.best_trial.number + 1})")
    print(f"Trial terburuk F1 : {min(trial_f1_values):.4f}")
    print(f"Mean F1 semua trial: {np.mean(trial_f1_values):.4f}")
    print(f"Std F1 semua trial : {np.std(trial_f1_values):.4f}")
    print(f"Rentang F1         : {max(trial_f1_values) - min(trial_f1_values):.4f}")

    # ── Convergence Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_numbers, trial_f1_values, c='steelblue', alpha=0.7,
               s=50, zorder=3, label='F1 per Trial')
    ax.plot(trial_numbers, best_so_far, c='crimson', linewidth=2,
            zorder=4, label='Best-so-far')
    ax.axhline(y=val_f1_default, color='gray', linestyle='--', linewidth=1,
               label=f'Default F1 = {val_f1_default:.4f}')
    ax.set_xlabel('Nomor Trial', fontsize=11)
    ax.set_ylabel('Macro F1-Score (Validasi)', fontsize=11)
    ax.set_title('Konvergensi Optimasi TPE — Macro F1-Score\n'
                 f'30 Trial, Best = {max(trial_f1_values):.4f} '
                 f'(Trial #{study_tpe.best_trial.number + 1})',
                 fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tpe_convergence_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Convergence chart disimpan: tpe_convergence_f1.png")


# ═══ CELL 12 ═══ Feature Importance (Gain-based, Top 20) ═══════════════════

print("\n" + "=" * 60)
print("CELL 12: Feature Importance (Gain-based, Top 20)")
print("=" * 60)

# ── Ambil feature importance dari model TPE-Optimized ────────────────────────
feat_imp = optimized_model.get_booster().get_score(importance_type='gain')
feat_imp_series = pd.Series(feat_imp).sort_values(ascending=False)

print("Top 20 Fitur berdasarkan Gain (TPE-Optimized):")
for rank, (feat, gain) in enumerate(feat_imp_series.head(20).items(), 1):
    bar = '█' * int(gain / feat_imp_series.iloc[0] * 30)
    print(f"  {rank:2d}. {feat:30s} {gain:>10.2f}  {bar}")

# ── Visualisasi Top 20 ────────────────────────────────────────────────────────
top20 = feat_imp_series.head(20)
fig, ax = plt.subplots(figsize=(10, 6))
top20.plot(kind='barh', ax=ax, color='teal', edgecolor='white')
ax.set_title('Top-20 Feature Importance (Gain)\nTPE-Optimized XGBoost — NF-UNSW-NB15-v3', fontsize=11)
ax.set_xlabel('Gain')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_gain_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Feature importance chart disimpan: feature_importance_gain_top20.png")


# ═══ CELL 13 ═══ Summary & Comparison Table ═════════════════════════════════

print("\n" + "=" * 60)
print("CELL 13: Summary & Comparison Table")
print("=" * 60)

# ── Ringkasan lengkap ─────────────────────────────────────────────────────────
print("\n" + "╔" + "═" * 66 + "╗")
print("║{:^66}║".format("PERBANDINGAN DEFAULT vs TPE-OPTIMIZED XGBOOST"))
print("║{:^66}║".format("Dataset: NF-UNSW-NB15-v3"))
print("╠" + "═" * 66 + "╣")
print("║{:<30}{:>18}{:>18}║".format("Metrik", "Default", "TPE-Optimized"))
print("╠" + "═" * 66 + "╣")

row_data = [
    ("Macro F1-Score",   f"{metrics['Default']['f1_macro']:.4f}",
                         f"{metrics['Optimized (TPE)']['f1_macro']:.4f}"),
    ("Accuracy",         f"{metrics['Default']['accuracy']:.4f}",
                         f"{metrics['Optimized (TPE)']['accuracy']:.4f}"),
    ("Cohen's Kappa",    f"{metrics['Default']['kappa']:.4f}",
                         f"{metrics['Optimized (TPE)']['kappa']:.4f}"),
]
for name, d_val, o_val in row_data:
    print("║{:<30}{:>18}{:>18}║".format("  " + name, d_val, o_val))

print("╠" + "═" * 66 + "╣")

delta_f1  = metrics['Optimized (TPE)']['f1_macro'] - metrics['Default']['f1_macro']
delta_acc = metrics['Optimized (TPE)']['accuracy']  - metrics['Default']['accuracy']
delta_kap = metrics['Optimized (TPE)']['kappa']     - metrics['Default']['kappa']

print("║{:<30}{:>18}{:>18}║".format("  ΔF1 (Opt - Default)",
      "", f"{delta_f1:+.4f}"))
print("║{:<30}{:>18}{:>18}║".format("  ΔAccuracy",
      "", f"{delta_acc:+.4f}"))
print("║{:<30}{:>18}{:>18}║".format("  ΔKappa",
      "", f"{delta_kap:+.4f}"))
print("╚" + "═" * 66 + "╝")

# ── Parameter comparison ──────────────────────────────────────────────────────
print("\n" + "╔" + "═" * 66 + "╗")
print("║{:^66}║".format("KONFIGURASI HIPERPARAMETER"))
print("╠" + "═" * 66 + "╣")
print("║{:<25}{:>20}{:>21}║".format("Parameter", "Default", "TPE-Optimized"))
print("╠" + "═" * 66 + "╣")
default_hp_disp = {
    'n_estimators': DEFAULT_PARAMS['n_estimators'],
    'learning_rate': DEFAULT_PARAMS['learning_rate'],
    'max_depth': DEFAULT_PARAMS['max_depth'],
    'min_child_weight': DEFAULT_PARAMS['min_child_weight'],
    'max_delta_step': DEFAULT_PARAMS['max_delta_step'],
    'gamma': DEFAULT_PARAMS['gamma'],
    'subsample': DEFAULT_PARAMS['subsample'],
    'colsample_bytree': DEFAULT_PARAMS['colsample_bytree'],
    'reg_alpha': DEFAULT_PARAMS['reg_alpha'],
    'reg_lambda': DEFAULT_PARAMS['reg_lambda'],
}
for k in default_hp_disp:
    d_v = str(default_hp_disp[k])
    o_v = str(round(study_tpe.best_trial.params.get(k, 'N/A'), 4))
    print("║{:<25}{:>20}{:>21}║".format("  " + k, d_v, o_v))
print("╚" + "═" * 66 + "╝")

# ── HP Importance Top 3 ───────────────────────────────────────────────────────
print("\nTop 3 HP Importance terhadap F1:")
for rank, (hp, imp) in enumerate(hp_importance.head(3).items(), 1):
    print(f"  {rank}. {hp}: {imp:.4f}")

# ── Optimization history ringkasan ────────────────────────────────────────────
all_f1_values = [t.value for t in study_tpe.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
print(f"\nRiwayat Optimasi TPE ({N_TRIALS} trial):")
print(f"  F1 Trial Pertama : {all_f1_values[0]:.4f}")
print(f"  F1 Terbaik       : {max(all_f1_values):.4f} (Trial #{study_tpe.best_trial.number})")
print(f"  F1 Terburuk      : {min(all_f1_values):.4f}")
print(f"  Mean F1 seluruh trial: {np.mean(all_f1_values):.4f}")

print("\n" + "=" * 60)
print("✅ SEMUA CELL SELESAI DIJALANKAN")
print("=" * 60)
print("Output yang dihasilkan:")
print("  - confusion_matrix_default_vs_optimized.png")
print("  - hp_importance_tpe_f1.png")
print("  - tpe_convergence_f1.png")
print("  - feature_importance_gain_top20.png")
