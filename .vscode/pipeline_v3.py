# ============================================================================
# PIPELINE V3 - BLOCK 1. IMPORTS & BASIC CONFIG
# ============================================================================
# 목적:
# walk-forward 검증용 환경 설정
# - 기존 feature 철학 유지
# - 룩어헤드 제거
# - HMM soft probability mapping 적용

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from hmmlearn.hmm import GaussianHMM

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

print("✓ pipeline_v3 started")
print("✓ Today:", date.today())

# ============================================================================
# PIPELINE V3 - BLOCK 2. LOAD RAW DATA
# ============================================================================
# 목적:
# raw KOSPI + VKOSPI 데이터를 불러온다.
# 이후 feature 생성은 전체 시계열 기준으로 하되,
# clipping / scaling / modeling은 fold별 train만 사용한다.

TICKER = "^KS11"
START_DATE = "2000-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

VKOSPI_CSV_PATH = "/Users/snu/regime-project/data/VKOSPI.csv"

print(f"Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
kospi = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

price_df = kospi[["Close"]].copy()
price_df.columns = ["price"]
price_df = price_df.dropna().sort_index()

print("✓ KOSPI loaded")
print(price_df.head())
print(price_df.tail())

# VKOSPI load
vkospi_raw = pd.read_csv(VKOSPI_CSV_PATH)

# 첫 번째 열: 날짜, 두 번째 열: 평균내재변동성_전체
vkospi_df = vkospi_raw.iloc[:, [0, 1]].copy()
vkospi_df.columns = ["date", "vkospi_level_raw"]

vkospi_df["date"] = pd.to_datetime(vkospi_df["date"], errors="coerce")
vkospi_df["vkospi_level_raw"] = pd.to_numeric(vkospi_df["vkospi_level_raw"], errors="coerce")
vkospi_df = vkospi_df.dropna().sort_values("date").set_index("date")

print("✓ VKOSPI loaded")
print(vkospi_df.head())
print(vkospi_df.tail())

# merge
price_df = price_df.join(vkospi_df, how="left")
price_df["vkospi_level_raw"] = price_df["vkospi_level_raw"].ffill()

print("✓ price_df merged")
print(price_df.head())
print(price_df.tail())
print(price_df[["price", "vkospi_level_raw"]].describe().round(4))

# ============================================================================
# PIPELINE V3 - BLOCK 3. FEATURE ENGINEERING
# ============================================================================
# 목적:
# 기존 연구와 동일한 7개 feature 생성
# - clipping 없음
# - train/test 분리는 이후 walk-forward에서 수행

def rolling_max_drawdown(price_series: pd.Series, window: int = 126) -> pd.Series:
    rolling_peak = price_series.rolling(window=window).max()
    drawdown = (price_series / rolling_peak) - 1.0
    rolling_mdd = drawdown.rolling(window=window).min()
    return rolling_mdd


def create_features(df_price: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df_price.index)

    # KOSPI 기반
    log_ret = np.log(df_price["price"] / df_price["price"].shift(1))
    features["ret_1m"] = df_price["price"].pct_change(21)
    features["ret_3m"] = df_price["price"].pct_change(63)
    features["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)
    features["mdd_6m"] = rolling_max_drawdown(df_price["price"], window=126)

    ma_60 = df_price["price"].rolling(60).mean()
    features["ma_gap_60"] = (df_price["price"] / ma_60) - 1.0

    # VKOSPI 기반
    vk = df_price["vkospi_level_raw"].copy()
    vk_log = np.log(vk)

    vk_mean_6m = vk_log.rolling(126).mean()
    vk_std_6m = vk_log.rolling(126).std()
    features["vkospi_z_6m"] = (vk_log - vk_mean_6m) / vk_std_6m

    features["vkospi_change_5d"] = vk_log.diff(5)

    return features


features_df = create_features(price_df)
print("✓ Features created")
print(features_df.head())
print(features_df.describe().round(4))

# ============================================================================
# PIPELINE V3 - BLOCK 4. FORWARD RETURN & EVENT LABEL
# ============================================================================
# 목적:
# 63일 / 126일 forward return 및 tail event label 생성
# target:
#   tail_event_63d = 1 if 63d_return < -0.15 else 0

def add_forward_returns_and_labels(df_price: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    price_series = df_price["price"]

    df["63d_return"] = price_series.shift(-63) / price_series - 1.0
    df["126d_return"] = price_series.shift(-126) / price_series - 1.0

    df["tail_event_63d"] = (df["63d_return"] < -0.15).astype(int)

    return df


full_df = add_forward_returns_and_labels(price_df, features_df)
full_df = full_df.dropna().copy()

FEATURE_COLS = [
    "ret_1m",
    "ret_3m",
    "vol_3m",
    "mdd_6m",
    "ma_gap_60",
    "vkospi_z_6m",
    "vkospi_change_5d"
]

TARGET_COL = "tail_event_63d"
FORWARD_RET_COL = "63d_return"

print("✓ Forward returns and event labels added")
print(full_df[[FORWARD_RET_COL, TARGET_COL]].describe().round(4))
print(full_df[[TARGET_COL]].mean().rename("event_rate"))
print(full_df.head())

# ============================================================================
# PIPELINE V3 - BLOCK 5. WALK-FORWARD SPLIT GENERATOR
# ============================================================================
# 목적:
# expanding-window walk-forward split 생성
# - 초기 8년 train
# - 이후 1년 test
# - 1년씩 이동

def generate_walkforward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 8,
    test_years: int = 1
):
    years = sorted(df.index.year.unique())
    splits = []

    for i in range(initial_train_years, len(years) - test_years + 1):
        train_years = years[:i]
        test_years_list = years[i:i + test_years]

        train_mask = df.index.year.isin(train_years)
        test_mask = df.index.year.isin(test_years_list)

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append({
            "fold": len(splits) + 1,
            "train_start": train_idx.min(),
            "train_end": train_idx.max(),
            "test_start": test_idx.min(),
            "test_end": test_idx.max(),
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    return splits


wf_splits = generate_walkforward_splits(full_df, initial_train_years=8, test_years=1)

print("✓ Walk-forward splits created")
print(f"Number of folds: {len(wf_splits)}")

for s in wf_splits[:5]:
    print(
        f"Fold {s['fold']}: "
        f"train {s['train_start'].date()} ~ {s['train_end'].date()} | "
        f"test {s['test_start'].date()} ~ {s['test_end'].date()}"
    )

# ============================================================================
# PIPELINE V3 - BLOCK 6. FOLD PREPROCESSING HELPERS
# ============================================================================
# 목적:
# 각 fold에서
# - clipping 경계는 train에서만 계산
# - scaler는 train에서만 fit
# - test는 transform만 수행

def fit_train_clipping_bounds(train_df: pd.DataFrame, feature_cols, lower_q=0.01, upper_q=0.99):
    bounds = {}
    for col in feature_cols:
        lower = train_df[col].quantile(lower_q)
        upper = train_df[col].quantile(upper_q)
        bounds[col] = (lower, upper)
    return bounds


def apply_clipping(df: pd.DataFrame, bounds: dict, feature_cols):
    clipped = df.copy()
    for col in feature_cols:
        lower, upper = bounds[col]
        clipped[col] = clipped[col].clip(lower, upper)
    return clipped


def preprocess_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    # clipping bounds from train only
    bounds = fit_train_clipping_bounds(train_df, feature_cols)

    train_clip = apply_clipping(train_df, bounds, feature_cols)
    test_clip = apply_clipping(test_df, bounds, feature_cols)

    # scaler fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clip[feature_cols])
    X_test = scaler.transform(test_clip[feature_cols])

    return X_train, X_test, train_clip, test_clip, scaler, bounds

# ============================================================================
# PIPELINE V3 - BLOCK 7. MODEL HELPERS
# ============================================================================
# 목적:
# KMeans / HMM 학습
# state -> event rate mapping
# HMM soft probability mapping

def fit_kmeans(X_train, n_clusters=4, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    train_states = model.fit_predict(X_train)
    return model, train_states


def predict_kmeans_states(model, X_test):
    return model.predict(X_test)


def fit_hmm(X_train, n_states=4, random_state=42):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=random_state
    )
    model.fit(X_train)
    train_states = model.predict(X_train)
    return model, train_states


def predict_hmm_states(model, X_test):
    return model.predict(X_test)


def predict_hmm_state_proba(model, X_test):
    return model.predict_proba(X_test)


def build_state_event_rate_map(states, y_event):
    temp = pd.DataFrame({
        "state": states,
        "event": y_event
    })

    state_event_rate = temp.groupby("state")["event"].mean().to_dict()
    state_count = temp.groupby("state")["event"].size().to_dict()

    return state_event_rate, state_count


def hard_state_probability(states_pred, state_event_rate_map, default_rate):
    probs = []
    for s in states_pred:
        probs.append(state_event_rate_map.get(s, default_rate))
    return np.array(probs)


def soft_state_probability(state_proba, state_event_rate_map, default_rate):
    """
    HMM soft mapping:
    p(event|x) = sum_s p(state=s|x) * p(event|state=s, train)
    """
    n_states = state_proba.shape[1]
    state_rates = np.array([
        state_event_rate_map.get(s, default_rate) for s in range(n_states)
    ])
    probs = state_proba @ state_rates
    return probs


def safe_log_loss(y_true, y_prob):
    # log_loss 안정성 위해 확률 clipping
    eps = 1e-6
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return log_loss(y_true, y_prob, labels=[0, 1])

# ============================================================================
# PIPELINE V3 - BLOCK 8. WALK-FORWARD MAIN LOOP
# ============================================================================
# 목적:
# 각 fold마다
# - train-only preprocessing
# - KMeans / HMM 학습
# - state-event mapping
# - out-of-sample prediction
# - metrics 계산

all_predictions = []
all_metrics = []
all_state_maps = []

BASE_EVENT_RATE = full_df[TARGET_COL].mean()

for split in wf_splits:
    fold = split["fold"]

    train_df = full_df.loc[split["train_idx"]].copy()
    test_df = full_df.loc[split["test_idx"]].copy()

    X_train, X_test, train_clip, test_clip, scaler, bounds = preprocess_fold(
        train_df, test_df, FEATURE_COLS
    )

    y_train = train_clip[TARGET_COL].values
    y_test = test_clip[TARGET_COL].values

    fold_event_rate = y_train.mean()

    # --------------------------------------------------
    # KMeans
    # --------------------------------------------------
    kmeans_model, kmeans_train_states = fit_kmeans(X_train, n_clusters=4)
    kmeans_test_states = predict_kmeans_states(kmeans_model, X_test)

    kmeans_state_event_rate, kmeans_state_count = build_state_event_rate_map(
        kmeans_train_states, y_train
    )

    kmeans_test_prob = hard_state_probability(
        kmeans_test_states,
        kmeans_state_event_rate,
        default_rate=fold_event_rate
    )

    # --------------------------------------------------
    # HMM
    # --------------------------------------------------
    hmm_model, hmm_train_states = fit_hmm(X_train, n_states=4)
    hmm_test_states = predict_hmm_states(hmm_model, X_test)
    hmm_test_state_proba = predict_hmm_state_proba(hmm_model, X_test)

    hmm_state_event_rate, hmm_state_count = build_state_event_rate_map(
        hmm_train_states, y_train
    )

    hmm_test_prob_hard = hard_state_probability(
        hmm_test_states,
        hmm_state_event_rate,
        default_rate=fold_event_rate
    )

    hmm_test_prob_soft = soft_state_probability(
        hmm_test_state_proba,
        hmm_state_event_rate,
        default_rate=fold_event_rate
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    model_prob_dict = {
        "kmeans_hard": kmeans_test_prob,
        "hmm_hard": hmm_test_prob_hard,
        "hmm_soft": hmm_test_prob_soft
    }

    for model_name, y_prob in model_prob_dict.items():
        # AUC는 y_test가 한 클래스만 있으면 계산 불가
        try:
            auc_val = roc_auc_score(y_test, y_prob)
        except:
            auc_val = np.nan

        brier_val = brier_score_loss(y_test, y_prob)
        logloss_val = safe_log_loss(y_test, y_prob)

        all_metrics.append({
            "fold": fold,
            "model": model_name,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_event_rate": y_train.mean(),
            "test_event_rate": y_test.mean(),
            "auc": auc_val,
            "brier": brier_val,
            "logloss": logloss_val
        })

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------
    pred_df = pd.DataFrame({
        "date": test_clip.index,
        "fold": fold,
        "y_true": y_test,
        "kmeans_state": kmeans_test_states,
        "hmm_state": hmm_test_states,
        "kmeans_prob": kmeans_test_prob,
        "hmm_prob_hard": hmm_test_prob_hard,
        "hmm_prob_soft": hmm_test_prob_soft
    })
    all_predictions.append(pred_df)

    # --------------------------------------------------
    # Save state-event maps
    # --------------------------------------------------
    for s, rate in kmeans_state_event_rate.items():
        all_state_maps.append({
            "fold": fold,
            "model": "kmeans",
            "state": s,
            "train_event_rate": rate,
            "train_state_count": kmeans_state_count.get(s, np.nan)
        })

    for s, rate in hmm_state_event_rate.items():
        all_state_maps.append({
            "fold": fold,
            "model": "hmm",
            "state": s,
            "train_event_rate": rate,
            "train_state_count": hmm_state_count.get(s, np.nan)
        })

    print(
        f"✓ Fold {fold} done | "
        f"train {split['train_start'].date()}~{split['train_end'].date()} | "
        f"test {split['test_start'].date()}~{split['test_end'].date()}"
    )

# ============================================================================
# PIPELINE V3 - BLOCK 9. BUILD OUTPUT TABLES
# ============================================================================
# 목적:
# prediction / metrics / state map 데이터프레임 정리

walkforward_predictions = pd.concat(all_predictions, axis=0).reset_index(drop=True)
walkforward_metrics = pd.DataFrame(all_metrics)
fold_state_event_map = pd.DataFrame(all_state_maps)

print("✓ Output tables built")
print("Predictions shape:", walkforward_predictions.shape)
print("Metrics shape:", walkforward_metrics.shape)
print("State-event map shape:", fold_state_event_map.shape)

print("\nMetrics preview:")
print(walkforward_metrics.head())

print("\nPredictions preview:")
print(walkforward_predictions.head())

print("\nState-event map preview:")
print(fold_state_event_map.head())

# ============================================================================
# PIPELINE V3 - BLOCK 10. SUMMARY METRICS
# ============================================================================
# 목적:
# 모델별 평균 성능 요약

summary_metrics = walkforward_metrics.groupby("model")[["auc", "brier", "logloss"]].agg(
    ["mean", "std", "median"]
)

print("✓ Summary metrics")
print(summary_metrics.round(4))

# ============================================================================
# PIPELINE V3 - BLOCK 11. SAVE OUTPUTS
# ============================================================================

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

walkforward_metrics.to_csv(os.path.join(SAVE_DIR, "walkforward_metrics.csv"), index=False)
walkforward_predictions.to_csv(os.path.join(SAVE_DIR, "walkforward_predictions.csv"), index=False)
fold_state_event_map.to_csv(os.path.join(SAVE_DIR, "fold_state_event_map.csv"), index=False)

print("✓ Outputs saved")
print("- outputs/walkforward_metrics.csv")
print("- outputs/walkforward_predictions.csv")
print("- outputs/fold_state_event_map.csv")

# ============================================================================
# PIPELINE V3 - BLOCK 12. VISUALIZATION
# ============================================================================
# 목적:
# 모델별 성능 시각화

plt.figure(figsize=(10, 6))
sns.boxplot(data=walkforward_metrics, x="model", y="auc")
plt.title("Walk-forward AUC by Model")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=walkforward_metrics, x="model", y="brier")
plt.title("Walk-forward Brier Score by Model")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=walkforward_metrics, x="model", y="logloss")
plt.title("Walk-forward LogLoss by Model")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V3 - BLOCK 13. OPTIONAL BASELINE COMPARISON
# ============================================================================
# 목적:
# baseline(상수 event rate)와 비교

baseline_metrics = []

for split in wf_splits:
    fold = split["fold"]

    train_df = full_df.loc[split["train_idx"]].copy()
    test_df = full_df.loc[split["test_idx"]].copy()

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    baseline_prob = np.repeat(y_train.mean(), len(y_test))

    try:
        auc_val = roc_auc_score(y_test, baseline_prob)
    except:
        auc_val = np.nan

    brier_val = brier_score_loss(y_test, baseline_prob)
    logloss_val = safe_log_loss(y_test, baseline_prob)

    baseline_metrics.append({
        "fold": fold,
        "model": "baseline_rate",
        "auc": auc_val,
        "brier": brier_val,
        "logloss": logloss_val
    })

baseline_metrics_df = pd.DataFrame(baseline_metrics)

print("✓ Baseline metrics")
print(baseline_metrics_df.head())

baseline_summary = baseline_metrics_df.groupby("model")[["auc", "brier", "logloss"]].agg(["mean", "std"])
print(baseline_summary.round(4))

# ============================================================================
# PIPELINE V3 - BLOCK 14. LOAD SAVED OUTPUTS FOR VISUALIZATION
# ============================================================================
# 목적:
# walk-forward prediction 결과와 state-event mapping 결과를 불러온다.

walkforward_predictions = pd.read_csv(
    "outputs/walkforward_predictions.csv",
    parse_dates=["date"]
)

fold_state_event_map = pd.read_csv(
    "outputs/fold_state_event_map.csv"
)

print("✓ Loaded walkforward outputs")
print("Predictions shape:", walkforward_predictions.shape)
print("State-event map shape:", fold_state_event_map.shape)

print("\nPredictions preview:")
print(walkforward_predictions.head())

print("\nState-event map preview:")
print(fold_state_event_map.head())

# ============================================================================
# PIPELINE V3 - BLOCK 15. RELABEL STATES BY RISK RANK WITHIN EACH FOLD
# ============================================================================
# 목적:
# fold마다 state 번호가 임의적이므로,
# train_event_rate 기준으로 state를 risk rank로 재정렬한다.
#
# 해석:
# risk_rank = 0  -> 가장 낮은 tail risk state
# risk_rank = 3  -> 가장 높은 tail risk state
# ============================================================================

def build_state_rank_map(fold_state_event_map, model_name):
    temp = fold_state_event_map[fold_state_event_map["model"] == model_name].copy()

    rank_records = []

    for fold in sorted(temp["fold"].unique()):
        fold_temp = temp[temp["fold"] == fold].copy()
        fold_temp = fold_temp.sort_values("train_event_rate").reset_index(drop=True)
        fold_temp["risk_rank"] = range(len(fold_temp))

        rank_records.append(fold_temp[["fold", "state", "risk_rank", "train_event_rate"]])

    return pd.concat(rank_records, axis=0).reset_index(drop=True)

kmeans_rank_map = build_state_rank_map(fold_state_event_map, "kmeans")
hmm_rank_map = build_state_rank_map(fold_state_event_map, "hmm")

print("✓ State rank maps created")

print("\nKMeans rank map preview:")
print(kmeans_rank_map.head())

print("\nHMM rank map preview:")
print(hmm_rank_map.head())

# ============================================================================
# PIPELINE V3 - BLOCK 16. MERGE RISK-RANKED STATES INTO PREDICTIONS
# ============================================================================
# 목적:
# walk-forward predictions에 fold별 risk_rank를 붙인다.
# ============================================================================

# KMeans
walkforward_predictions = walkforward_predictions.merge(
    kmeans_rank_map.rename(columns={"state": "kmeans_state", "risk_rank": "kmeans_risk_rank"}),
    on=["fold", "kmeans_state"],
    how="left"
)

# HMM
walkforward_predictions = walkforward_predictions.merge(
    hmm_rank_map.rename(columns={"state": "hmm_state", "risk_rank": "hmm_risk_rank"}),
    on=["fold", "hmm_state"],
    how="left"
)

print("✓ Risk-ranked states merged into predictions")
print(walkforward_predictions.head())

# ============================================================================
# PIPELINE V3 - BLOCK 17. MERGE PREDICTIONS WITH PRICE DATA
# ============================================================================
# 목적:
# OOS predicted state를 가격 위에 표시하기 위해 KOSPI 가격을 붙인다.
# ============================================================================

price_plot_df = price_df[["price"]].copy().reset_index()
price_plot_df.columns = ["date", "price"]

v3_plot_df = walkforward_predictions.merge(
    price_plot_df,
    on="date",
    how="left"
).sort_values("date").reset_index(drop=True)

print("✓ Plot dataframe ready")
print(v3_plot_df.head())
print(v3_plot_df.tail())

# ============================================================================
# PIPELINE V3 - BLOCK 18. OOS REGIME TRANSITION PLOTS (V1-STYLE)
# ============================================================================
# 목적:
# out-of-sample predicted state를 가격 위에 표시한다.
# state 번호 대신 fold-aligned risk_rank 사용
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 11))

# ----------------------------------------------------------------------
# 1. Price with KMeans OOS predicted states
# ----------------------------------------------------------------------
ax = axes[0]
for r in sorted(v3_plot_df["kmeans_risk_rank"].dropna().unique()):
    mask = v3_plot_df["kmeans_risk_rank"] == r
    ax.scatter(
        v3_plot_df.loc[mask, "date"],
        v3_plot_df.loc[mask, "price"],
        label=f"Risk Rank {int(r)}",
        alpha=0.6,
        s=20
    )

ax.plot(v3_plot_df["date"], v3_plot_df["price"], color="black", alpha=0.25, linewidth=1)
ax.set_title("Walk-forward OOS KMeans Regime Predictions", fontweight="bold")
ax.set_ylabel("KOSPI Price")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# ----------------------------------------------------------------------
# 2. Price with HMM OOS predicted states
# ----------------------------------------------------------------------
ax = axes[1]
for r in sorted(v3_plot_df["hmm_risk_rank"].dropna().unique()):
    mask = v3_plot_df["hmm_risk_rank"] == r
    ax.scatter(
        v3_plot_df.loc[mask, "date"],
        v3_plot_df.loc[mask, "price"],
        label=f"Risk Rank {int(r)}",
        alpha=0.6,
        s=20
    )

ax.plot(v3_plot_df["date"], v3_plot_df["price"], color="black", alpha=0.25, linewidth=1)
ax.set_title("Walk-forward OOS HMM Regime Predictions", fontweight="bold")
ax.set_ylabel("KOSPI Price")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# ----------------------------------------------------------------------
# 3. Recent 2-year state history
# ----------------------------------------------------------------------
recent_df = v3_plot_df[v3_plot_df["date"] >= (v3_plot_df["date"].max() - pd.DateOffset(years=2))].copy()

ax = axes[2]
ax.step(
    recent_df["date"],
    recent_df["kmeans_risk_rank"],
    where="mid",
    alpha=0.7,
    label="KMeans risk rank"
)
ax.plot(
    recent_df["date"],
    recent_df["hmm_risk_rank"],
    marker="o",
    markersize=3,
    linewidth=1.2,
    alpha=0.8,
    label="HMM risk rank"
)

ax.set_title("Recent 2-Year OOS Regime History", fontweight="bold")
ax.set_ylabel("Risk Rank")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V3 - BLOCK 19. RECENT OOS REGIME STATUS CHECK
# ============================================================================
# 목적:
# 최근 20일 / 60일 OOS predicted regime 분포를 확인한다.
# ============================================================================

recent_20 = v3_plot_df.tail(20)
recent_60 = v3_plot_df.tail(60)

print("=" * 80)
print("V3 OOS CURRENT REGIME STATUS")
print("=" * 80)

print("\n--- Last 20 OOS Trading Days ---")
print("KMeans risk-rank distribution:")
print(recent_20["kmeans_risk_rank"].value_counts().sort_index())
print("Most common KMeans risk rank:", recent_20["kmeans_risk_rank"].mode().iloc[0])

print("\nHMM risk-rank distribution:")
print(recent_20["hmm_risk_rank"].value_counts().sort_index())
print("Most common HMM risk rank:", recent_20["hmm_risk_rank"].mode().iloc[0])

print("\n--- Last 60 OOS Trading Days ---")
print("KMeans risk-rank distribution:")
print(recent_60["kmeans_risk_rank"].value_counts().sort_index())

print("\nHMM risk-rank distribution:")
print(recent_60["hmm_risk_rank"].value_counts().sort_index())

print("\nLatest OOS date:", v3_plot_df["date"].iloc[-1].date())
print("Latest KMeans risk rank:", v3_plot_df["kmeans_risk_rank"].iloc[-1])
print("Latest HMM risk rank:", v3_plot_df["hmm_risk_rank"].iloc[-1])

# ============================================================================
# PIPELINE V3 - BLOCK 20. SAVE PLOT-READY DATA
# ============================================================================
# 목적:
# plot에 사용한 OOS regime dataframe 저장
# ============================================================================

v3_plot_df.to_csv("outputs/walkforward_regime_plot_df.csv", index=False)
print("✓ Saved: outputs/walkforward_regime_plot_df.csv")

# ============================================================================
# LAST BLOCK. SAVE FINAL TABLES / JSON / FIGURES
# ============================================================================
# 목적:
# - 핵심 결과표 저장
# - 현재 상태 요약 JSON 저장
# - 주요 시각화 PNG 저장
# - 보고서에 바로 넣을 수 있는 최소 결과물 정리
# ============================================================================

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. SAVE PATHS
# ----------------------------------------------------------------------
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

print("✓ Output directories ready")


# ----------------------------------------------------------------------
# 1. SAVE CORE TABLES
# ----------------------------------------------------------------------
# 존재하는 변수만 저장
table_save_list = {
    "v2_payoff_summary.csv": "payoff_summary_df",
    "v2_scenario_summary.csv": "scenario_summary_df",
    "v2_comparison_current_vs_transition.csv": "comparison_df",
    "v2_state_contribution.csv": "contribution_df",
    "v3_walkforward_metrics.csv": "walkforward_metrics",
    "v3_walkforward_predictions.csv": "walkforward_predictions",
    "v3_fold_state_event_map.csv": "fold_state_event_map",
    "v3_regime_plot_df.csv": "v3_plot_df",
}

for fname, varname in table_save_list.items():
    if varname in globals():
        obj = globals()[varname]
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(os.path.join(TABLE_DIR, fname), index=True)
            print(f"✓ Saved table: {fname}")


# ----------------------------------------------------------------------
# 2. SAVE SUMMARY METRIC TABLES
# ----------------------------------------------------------------------
if "walkforward_metrics" in globals():
    v3_summary_metrics = walkforward_metrics.groupby("model")[["auc", "brier", "logloss"]].agg(
        ["mean", "std", "median"]
    )
    v3_summary_metrics.to_csv(os.path.join(TABLE_DIR, "v3_summary_metrics.csv"))
    print("✓ Saved table: v3_summary_metrics.csv")

if "baseline_metrics_df" in globals():
    baseline_summary = baseline_metrics_df.groupby("model")[["auc", "brier", "logloss"]].agg(
        ["mean", "std", "median"]
    )
    baseline_summary.to_csv(os.path.join(TABLE_DIR, "v3_baseline_summary.csv"))
    print("✓ Saved table: v3_baseline_summary.csv")


# ----------------------------------------------------------------------
# 3. SAVE JSON SUMMARY
# ----------------------------------------------------------------------
final_summary = {}

# v1 / v2 current state info
if "current_hmm_state" in globals():
    final_summary["current_hmm_state"] = int(current_hmm_state)

if "current_kmeans_regime" in globals():
    final_summary["current_kmeans_regime"] = int(current_kmeans_regime)

if "comparison_df" in globals():
    try:
        final_summary["current_vs_transition_weighted"] = {
            idx: {
                "current_state_mean": float(comparison_df.loc[idx, "current_state_mean"]),
                "transition_weighted_mean_payoff": float(comparison_df.loc[idx, "transition_weighted_mean_payoff"])
            }
            for idx in comparison_df.index
        }
    except Exception:
        pass

if "scenario_mean_table" in globals():
    try:
        final_summary["scenario_mean_payoff"] = {
            str(idx): {str(col): float(scenario_mean_table.loc[idx, col]) for col in scenario_mean_table.columns}
            for idx in scenario_mean_table.index
        }
    except Exception:
        pass

if "scenario_negative_table" in globals():
    try:
        final_summary["scenario_negative_payoff_probability"] = {
            str(idx): {str(col): float(scenario_negative_table.loc[idx, col]) for col in scenario_negative_table.columns}
            for idx in scenario_negative_table.index
        }
    except Exception:
        pass

if "walkforward_metrics" in globals():
    try:
        wf_summary = walkforward_metrics.groupby("model")[["auc", "brier", "logloss"]].mean().round(6)
        final_summary["walkforward_mean_metrics"] = {
            str(idx): {str(col): float(wf_summary.loc[idx, col]) for col in wf_summary.columns}
            for idx in wf_summary.index
        }
    except Exception:
        pass

if "baseline_metrics_df" in globals():
    try:
        base_summary = baseline_metrics_df.groupby("model")[["auc", "brier", "logloss"]].mean().round(6)
        final_summary["baseline_mean_metrics"] = {
            str(idx): {str(col): float(base_summary.loc[idx, col]) for col in base_summary.columns}
            for idx in base_summary.index
        }
    except Exception:
        pass

if "v3_plot_df" in globals() and len(v3_plot_df) > 0:
    try:
        recent_20 = v3_plot_df.tail(20)
        recent_60 = v3_plot_df.tail(60)

        final_summary["v3_recent_regime_status"] = {
            "latest_oos_date": str(v3_plot_df["date"].iloc[-1].date()),
            "latest_kmeans_risk_rank": None if pd.isna(v3_plot_df["kmeans_risk_rank"].iloc[-1]) else int(v3_plot_df["kmeans_risk_rank"].iloc[-1]),
            "latest_hmm_risk_rank": None if pd.isna(v3_plot_df["hmm_risk_rank"].iloc[-1]) else int(v3_plot_df["hmm_risk_rank"].iloc[-1]),
            "recent_20_kmeans_mode": None if recent_20["kmeans_risk_rank"].dropna().empty else int(recent_20["kmeans_risk_rank"].mode().iloc[0]),
            "recent_20_hmm_mode": None if recent_20["hmm_risk_rank"].dropna().empty else int(recent_20["hmm_risk_rank"].mode().iloc[0]),
            "recent_60_kmeans_mode": None if recent_60["kmeans_risk_rank"].dropna().empty else int(recent_60["kmeans_risk_rank"].mode().iloc[0]),
            "recent_60_hmm_mode": None if recent_60["hmm_risk_rank"].dropna().empty else int(recent_60["hmm_risk_rank"].mode().iloc[0]),
        }
    except Exception:
        pass

with open(os.path.join(JSON_DIR, "final_summary.json"), "w", encoding="utf-8") as f:
    json.dump(final_summary, f, ensure_ascii=False, indent=2)

print("✓ Saved JSON: final_summary.json")


# ----------------------------------------------------------------------
# 4. SAVE FIGURES
# ----------------------------------------------------------------------
# 4-1. V2 Mean payoff by state
if "mean_payoff_table" in globals():
    ax = mean_payoff_table.plot(kind="bar", figsize=(12, 6))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Mean Payoff by HMM State")
    plt.xlabel("HMM State")
    plt.ylabel("Mean Payoff")
    plt.xticks(rotation=0)
    plt.legend(title="Product")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_v2_mean_payoff_by_state.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved figure: fig_v2_mean_payoff_by_state.png")

# 4-2. V2 Negative payoff probability
if "negative_prob_table" in globals():
    ax = negative_prob_table.plot(kind="bar", figsize=(12, 6))
    plt.title("Negative Payoff Probability by HMM State")
    plt.xlabel("HMM State")
    plt.ylabel("Probability of Negative Payoff")
    plt.xticks(rotation=0)
    plt.legend(title="Product")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_v2_negative_payoff_by_state.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved figure: fig_v2_negative_payoff_by_state.png")

# 4-3. V2 Current vs transition-weighted
if "comparison_df" in globals():
    comparison_plot_df = comparison_df.reset_index().rename(columns={"index": "product"})
    comparison_plot_df = comparison_plot_df.melt(
        id_vars="product",
        value_vars=["current_state_mean", "transition_weighted_mean_payoff"],
        var_name="scenario_type",
        value_name="mean_payoff"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_plot_df, x="product", y="mean_payoff", hue="scenario_type")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Current-State vs Transition-Weighted Expected Payoff (State {current_hmm_state})")
    plt.xlabel("Product")
    plt.ylabel("Mean Payoff")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_v2_current_vs_transition.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved figure: fig_v2_current_vs_transition.png")

# 4-4. V2 Scenario mean payoff
if "scenario_summary_df" in globals():
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scenario_summary_df, x="scenario", y="mean", hue="product")
    plt.title("Mean Payoff under Regime Evolution Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Mean Payoff")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_v2_scenario_mean_payoff.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved figure: fig_v2_scenario_mean_payoff.png")

# 4-5. V3 AUC / Brier / LogLoss
if "walkforward_metrics" in globals():
    for metric_name in ["auc", "brier", "logloss"]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=walkforward_metrics, x="model", y=metric_name)
        plt.title(f"Walk-forward {metric_name.upper()} by Model")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"fig_v3_{metric_name}_boxplot.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved figure: fig_v3_{metric_name}_boxplot.png")

# 4-6. V3 OOS regime plot
if "v3_plot_df" in globals() and len(v3_plot_df) > 0:
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))

    ax = axes[0]
    for r in sorted(v3_plot_df["kmeans_risk_rank"].dropna().unique()):
        mask = v3_plot_df["kmeans_risk_rank"] == r
        ax.scatter(
            v3_plot_df.loc[mask, "date"],
            v3_plot_df.loc[mask, "price"],
            label=f"Risk Rank {int(r)}",
            alpha=0.6,
            s=20
        )
    ax.plot(v3_plot_df["date"], v3_plot_df["price"], color="black", alpha=0.25, linewidth=1)
    ax.set_title("Walk-forward OOS KMeans Regime Predictions", fontweight="bold")
    ax.set_ylabel("KOSPI Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in sorted(v3_plot_df["hmm_risk_rank"].dropna().unique()):
        mask = v3_plot_df["hmm_risk_rank"] == r
        ax.scatter(
            v3_plot_df.loc[mask, "date"],
            v3_plot_df.loc[mask, "price"],
            label=f"Risk Rank {int(r)}",
            alpha=0.6,
            s=20
        )
    ax.plot(v3_plot_df["date"], v3_plot_df["price"], color="black", alpha=0.25, linewidth=1)
    ax.set_title("Walk-forward OOS HMM Regime Predictions", fontweight="bold")
    ax.set_ylabel("KOSPI Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    recent_df = v3_plot_df[v3_plot_df["date"] >= (v3_plot_df["date"].max() - pd.DateOffset(years=2))].copy()

    ax = axes[2]
    ax.step(recent_df["date"], recent_df["kmeans_risk_rank"], where="mid", alpha=0.7, label="KMeans risk rank")
    ax.plot(recent_df["date"], recent_df["hmm_risk_rank"], marker="o", markersize=3, linewidth=1.2, alpha=0.8, label="HMM risk rank")
    ax.set_title("Recent 2-Year OOS Regime History", fontweight="bold")
    ax.set_ylabel("Risk Rank")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_v3_oos_regime_transitions.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Saved figure: fig_v3_oos_regime_transitions.png")


# ----------------------------------------------------------------------
# 5. SIMPLE REPORT CHECK PRINT
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL SAVE SUMMARY")
print("=" * 80)
print(f"Tables saved to : {TABLE_DIR}")
print(f"JSON saved to   : {JSON_DIR}")
print(f"Figures saved to: {FIG_DIR}")

print("\nSaved files preview:")
print("- final_summary.json")
print("- v2_payoff_summary.csv")
print("- v2_scenario_summary.csv")
print("- v2_comparison_current_vs_transition.csv")
print("- v3_walkforward_metrics.csv")
print("- v3_walkforward_predictions.csv")
print("- v3_summary_metrics.csv")
print("- fig_v2_mean_payoff_by_state.png")
print("- fig_v2_current_vs_transition.png")
print("- fig_v2_scenario_mean_payoff.png")
print("- fig_v3_oos_regime_transitions.png")
print("- fig_v3_auc_boxplot.png / brier / logloss")
print("=" * 80)