# ============================================================================
# BLOCK 1. IMPORTS & BASIC CONFIG
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import date, datetime

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix

from hmmlearn.hmm import GaussianHMM
from scipy.stats import skew, kurtosis
from IPython.display import display

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

print("✓ Libraries imported")
print("✓ Today:", date.today())

# ============================================================================
# BLOCK 2. CONFIGURATION
# ============================================================================

TICKER = "^KS11"   # KOSPI Index
START_DATE = "2000-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

# Feature windows
VOL_WINDOW = 20
MOM_1M = 21
MOM_3M = 63

# Forward horizons
FWD_3M = 63
FWD_6M = 126

# Tail thresholds
LOSS_THRESHOLD = -0.15
TAIL_THRESHOLD = 0.20


print("Ticker:", TICKER)
print("Period:", START_DATE, "to", END_DATE)

# ============================================================================
# BLOCK 3. DATA DOWNLOAD
# ============================================================================

print(f"Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

price_df = raw[["Close"]].copy()
price_df.columns = ["price"]
price_df = price_df.dropna()

print(f"✓ Downloaded {len(price_df)} rows")
print(f"✓ Date range: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

display(price_df.head())
display(price_df.tail())

# ============================================================================
# BLOCK 3.5. LOAD VKOSPI DATA
# ============================================================================
# 목적:
# KRX에서 다운로드한 VKOSPI 데이터를 불러오고,
# 사용할 대표 컬럼만 남겨서 price_df와 병합 가능한 형태로 정리한다.

VKOSPI_CSV_PATH = "/Users/snu/regime-project/data/VKOSPI.csv"

vkospi_raw = pd.read_csv(VKOSPI_CSV_PATH)

print("Original VKOSPI columns:")
print(vkospi_raw.columns.tolist())

# --------------------------------------------------------------------------
# 사용할 컬럼 선택
# A열: 일자
# B열: 평균내재변동 (대표 VKOSPI level로 사용)
# --------------------------------------------------------------------------
vkospi_df = vkospi_raw.iloc[:, [0, 1]].copy()
vkospi_df.columns = ["date", "vkospi_level_raw"]

# 날짜 변환
vkospi_df["date"] = pd.to_datetime(vkospi_df["date"], errors="coerce")

# 숫자형 변환
vkospi_df["vkospi_level_raw"] = pd.to_numeric(vkospi_df["vkospi_level_raw"], errors="coerce")

# 결측 제거 및 정렬
vkospi_df = vkospi_df.dropna().sort_values("date")

# 인덱스 설정
vkospi_df = vkospi_df.set_index("date")

print("✓ VKOSPI raw series loaded")
print(vkospi_df.head())
print(vkospi_df.tail())
print(vkospi_df.describe().round(4))

# ============================================================================
# BLOCK 3.6. MERGE VKOSPI WITH PRICE DATA
# ============================================================================
# 목적:
# KOSPI 가격 데이터와 VKOSPI 데이터를 날짜 기준으로 병합한다.

price_df = price_df.join(vkospi_df, how="left")

# 휴장/결측은 직전 값으로 보정
price_df["vkospi_level_raw"] = price_df["vkospi_level_raw"].ffill()

print("✓ VKOSPI merged into price_df")
print(price_df[["price", "vkospi_level_raw"]].head())
print(price_df[["price", "vkospi_level_raw"]].tail())
print(price_df[["vkospi_level_raw"]].describe().round(4))

# ============================================================================
# BLOCK 4. PREPROCESSING FUNCTION
# ============================================================================

def preprocess_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for price series.
    - sort index
    - remove duplicates
    - ensure numeric price
    """
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"])
    return out

price_df = preprocess_price(price_df)

print("✓ Preprocessing completed")
print("Rows after preprocessing:", len(price_df))

# ============================================================================
# BLOCK 5. FEATURE ENGINEERING (FINAL)
# ============================================================================
# 목적:
# 현재 시장 상태(regime)를 설명할 수 있는 핵심 변수 5개를 생성한다.
#
# 사용 변수:
# 1. ret_1m      : 최근 1개월 수익률 (단기 방향성, simple return)
# 2. ret_3m      : 최근 3개월 수익률 (중기 방향성, simple return)
# 3. vol_3m      : 최근 3개월 실현변동성 (불확실성, log return 기반)
# 4. mdd_6m      : 최근 6개월 최대낙폭 (하락 스트레스)
# 5. ma_gap_60   : 60일 이동평균 괴리율 (과열/고점 근접도)
'''
def rolling_max_drawdown(price_series: pd.Series, window: int = 126) -> pd.Series:
    """
    최근 일정 기간(window) 내 최대낙폭(MDD)을 계산한다.

    Parameters
    ----------
    price_series : pd.Series
        가격 시계열
    window : int
        롤링 윈도우 길이 (기본값 126일 ≈ 6개월)

    Returns
    -------
    pd.Series
        각 시점에서 최근 window 동안의 최대낙폭
    """
    rolling_peak = price_series.rolling(window=window).max()
    drawdown = (price_series / rolling_peak) - 1.0
    rolling_mdd = drawdown.rolling(window=window).min()
    return rolling_mdd


def create_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    KOSPI 가격 데이터로부터 regime 분류용 feature를 생성한다.

    Parameters
    ----------
    df_price : pd.DataFrame
        'price' 컬럼을 가진 DataFrame

    Returns
    -------
    pd.DataFrame
        regime 분류용 feature DataFrame
    """
    features = pd.DataFrame(index=df_price.index)

    # 일별 log return (변동성 계산용)
    log_ret = np.log(df_price["price"] / df_price["price"].shift(1))

    # 1개월 수익률 (약 21거래일, simple return)
    features["ret_1m"] = df_price["price"].pct_change(21)

    # 3개월 수익률 (약 63거래일, simple return)
    features["ret_3m"] = df_price["price"].pct_change(63)

    # 3개월 실현변동성 (annualized, log return 기반)
    features["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)

    # 6개월 rolling 최대낙폭
    features["mdd_6m"] = rolling_max_drawdown(df_price["price"], window=126)

    # 60일 이동평균 괴리율
    ma_60 = df_price["price"].rolling(60).mean()
    features["ma_gap_60"] = (df_price["price"] / ma_60) - 1.0

    return features


# Feature 생성
features_df = create_features(price_df)

# NaN 제거
features_df = features_df.dropna()

print("✓ Features created successfully")
print(f"Number of observations: {len(features_df)}")
print("\nFeature columns:")
print(features_df.columns.tolist())

print("\nSummary statistics:")
print(features_df.describe().round(4))

print("\nFirst 5 rows:")
print(features_df.head())
'''

# ============================================================================
# BLOCK 5. FEATURE ENGINEERING (WITH REVISED VKOSPI FEATURES)
# ============================================================================
# 목적:
# KOSPI 가격 기반 feature + VKOSPI 기반 심리/불확실성 feature를 생성한다.
#
# 기존 KOSPI feature:
# - ret_1m
# - ret_3m
# - vol_3m
# - mdd_6m
# - ma_gap_60
#
# VKOSPI feature (revised):
# - vkospi_z_6m     : 최근 6개월 기준 상대적 불안 수준
# - vkospi_change_5d: 최근 5거래일 VKOSPI 변화량 (log diff)
#
# 주의:
# - VKOSPI raw level은 장기 시계열에서 레벨 차이와 위기 구간 극단값 영향이 커서
#   그대로 쓰지 않고 rolling z-score로 변환
# - 변화율은 1개월 대신 5거래일 변화로 단기 심리 변화를 반영
# ============================================================================

def rolling_max_drawdown(price_series: pd.Series, window: int = 126) -> pd.Series:
    """
    rolling max drawdown 계산
    """
    rolling_peak = price_series.rolling(window=window).max()
    drawdown = (price_series / rolling_peak) - 1.0
    rolling_mdd = drawdown.rolling(window=window).min()
    return rolling_mdd


def create_features(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    KOSPI + VKOSPI 기반 regime feature 생성
    """
    features = pd.DataFrame(index=df_price.index)

    # ----------------------------------------------------------------------
    # 1. KOSPI price-based features
    # ----------------------------------------------------------------------
    log_ret = np.log(df_price["price"] / df_price["price"].shift(1))

    # 1개월 수익률
    features["ret_1m"] = df_price["price"].pct_change(21)

    # 3개월 수익률
    features["ret_3m"] = df_price["price"].pct_change(63)

    # 3개월 연율화 변동성
    features["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)

    # 6개월 rolling MDD
    features["mdd_6m"] = rolling_max_drawdown(df_price["price"], window=126)

    # 60일 이동평균 괴리율
    ma_60 = df_price["price"].rolling(60).mean()
    features["ma_gap_60"] = (df_price["price"] / ma_60) - 1.0

    # ----------------------------------------------------------------------
    # 2. VKOSPI sentiment/uncertainty features (revised)
    # ----------------------------------------------------------------------
    vk = df_price["vkospi_level_raw"].copy()

    # 로그 변환
    vk_log = np.log(vk)

    # 6개월 rolling z-score
    vk_mean_6m = vk_log.rolling(126).mean()
    vk_std_6m = vk_log.rolling(126).std()
    features["vkospi_z_6m"] = (vk_log - vk_mean_6m) / vk_std_6m

    # 최근 5거래일 VKOSPI 변화량 (log diff)
    features["vkospi_change_5d"] = vk_log.diff(5)

    # ----------------------------------------------------------------------
    # 3. 결측 제거
    # ----------------------------------------------------------------------
    features = features.dropna()

    # ----------------------------------------------------------------------
    # 4. Extreme value clipping for VKOSPI features
    # ----------------------------------------------------------------------
    for col in ["vkospi_z_6m", "vkospi_change_5d"]:
        lower = features[col].quantile(0.01)
        upper = features[col].quantile(0.99)
        features[col] = features[col].clip(lower, upper)

    return features


# feature 생성
features_df = create_features(price_df)

print("✓ Features with revised VKOSPI created successfully")
print(f"Number of observations: {len(features_df)}")

print("\nFeature columns:")
print(features_df.columns.tolist())

print("\nSummary statistics:")
print(features_df.describe().round(4))

print("\nFirst 5 rows:")
print(features_df.head())
# ============================================================================
# BLOCK 5.5. FEATURE CORRELATION CHECK
# ============================================================================

feature_cols = [
    "ret_1m",
    "ret_3m",
    "vol_3m",
    "mdd_6m",
    "ma_gap_60",
    "vkospi_z_6m",
    "vkospi_change_5d"
]

corr_matrix = features_df[feature_cols].corr()

print("Feature Correlation Matrix:")
print(corr_matrix.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================================
# BLOCK 6. FEATURE SCALING
# ============================================================================
'''
feature_cols = [
    "ret_1m",
    "ret_3m",
    "vol_3m",
    "mdd_6m",
    "ma_gap_60"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df[feature_cols])

print("✓ Feature scaling completed")
print("Scaled shape:", X_scaled.shape)
print("Features used:", feature_cols)
'''
# ============================================================================
# BLOCK 6. FEATURE SCALING (REVISED VKOSPI)
# ============================================================================

feature_cols = [
    "ret_1m",
    "ret_3m",
    "vol_3m",
    "mdd_6m",
    "ma_gap_60",
    "vkospi_z_6m",
    "vkospi_change_5d"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df[feature_cols])

print("✓ Feature scaling completed")
print("Scaled shape:", X_scaled.shape)
print("Features used:", feature_cols)

# ============================================================================
# BLOCK 7. KMEANS REGIME CLASSIFICATION
# ============================================================================
'''
inertias = []
silhouette_scores = []
K_range = range(2, 9)

print("Testing KMeans for different K...")
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    print(f"K={k}: Inertia={model.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

optimal_k_kmeans = K_range[np.argmax(silhouette_scores)]
print("\n✓ Optimal K (KMeans):", optimal_k_kmeans)

kmeans_final = KMeans(n_clusters=optimal_k_kmeans, random_state=42, n_init=10)
features_df["regime_kmeans"] = kmeans_final.fit_predict(X_scaled)
'''

KMEANS_K = 4

kmeans_final = KMeans(
    n_clusters=KMEANS_K,
    random_state=42,
    n_init=20
)

features_df["regime_kmeans"] = kmeans_final.fit_predict(X_scaled)

print("✓ KMeans fitted successfully")
print(f"Fixed number of clusters: {KMEANS_K}")

print("\nKMeans regime distribution:")
print(features_df["regime_kmeans"].value_counts().sort_index())

# ============================================================================
# BLOCK 8. KMEANS MODEL SELECTION PLOT
# ============================================================================
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
ax1.axvline(optimal_k_kmeans, color="r", linestyle="--", label=f"Optimal K={optimal_k_kmeans}")
ax1.set_title("KMeans Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Inertia")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, "go-", linewidth=2, markersize=8)
ax2.axvline(optimal_k_kmeans, color="r", linestyle="--", label=f"Optimal K={optimal_k_kmeans}")
ax2.set_title("Silhouette Score")
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(features_df["regime_kmeans"].value_counts().sort_index())
'''

kmeans_profile = features_df.groupby("regime_kmeans")[feature_cols].mean().round(4)

print("KMeans regime feature means:")
print(kmeans_profile)
# ============================================================================
# BLOCK 9. HMM REGIME CLASSIFICATION
# ============================================================================
'''
hmm_models = {}
bic_scores = []
aic_scores = []
n_states_range = range(2, 7)

print("Training HMM models...")
for n_states in n_states_range:
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    hmm.fit(X_scaled)
    hmm_models[n_states] = hmm

    bic = hmm.bic(X_scaled)
    aic = hmm.aic(X_scaled)
    bic_scores.append(bic)
    aic_scores.append(aic)

    print(f"States={n_states}: BIC={bic:.2f}, AIC={aic:.2f}")

HMM_STATES = n_states_range[np.argmin(bic_scores)]
print("\n✓ Optimal HMM states:", HMM_STATES)

hmm_final = hmm_models[HMM_STATES]
features_df["regime_hmm"] = hmm_final.predict(X_scaled)
'''
HMM_STATES = 4

hmm_final = GaussianHMM(
    n_components=HMM_STATES,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)

hmm_final.fit(X_scaled)
features_df["regime_hmm"] = hmm_final.predict(X_scaled)

print("✓ HMM fitted successfully")
print(f"Fixed number of states: {HMM_STATES}")

print("\nHMM regime distribution:")
print(features_df["regime_hmm"].value_counts().sort_index())
# ============================================================================
# BLOCK 10. HMM MODEL SELECTION PLOT
# ============================================================================
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(n_states_range, bic_scores, "ro-", linewidth=2, markersize=8)
ax1.axvline(HMM_STATES, color="g", linestyle="--", label=f"Optimal={HMM_STATES}")
ax1.set_title("HMM Model Selection (BIC)")
ax1.set_xlabel("Number of States")
ax1.set_ylabel("BIC")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(n_states_range, aic_scores, "bo-", linewidth=2, markersize=8)
ax2.axvline(HMM_STATES, color="g", linestyle="--", label=f"Optimal={HMM_STATES}")
ax2.set_title("HMM Model Selection (AIC)")
ax2.set_xlabel("Number of States")
ax2.set_ylabel("AIC")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(features_df["regime_hmm"].value_counts().sort_index())
'''
# ============================================================================
# BLOCK 10. HMM REGIME PROFILE & TRANSITION MATRIX
# ============================================================================

hmm_profile = features_df.groupby("regime_hmm")[feature_cols].mean().round(4)

print("HMM state feature means:")
print(hmm_profile)

print("\nHMM transition matrix:")
transmat_df = pd.DataFrame(
    hmm_final.transmat_,
    index=[f"State_{i}" for i in range(HMM_STATES)],
    columns=[f"State_{i}" for i in range(HMM_STATES)]
).round(4)

print(transmat_df)
features_df["regime_hmm"].value_counts(normalize=True)

# ============================================================================
# BLOCK 10.5. HMM REGIME DIAGNOSTIC CHECK
# ============================================================================
# 목적:
# 1) 최근 60거래일 동안 현재 시장 state가 어떻게 분포하는지 확인
# 2) 각 state가 평균적으로 얼마나 오래 지속되는지 확인

# 최근 60거래일 state 분포
recent_60_states = features_df["regime_hmm"].tail(60)

print("=" * 80)
print("HMM RECENT 60-DAY STATE CHECK")
print("=" * 80)

print("\nRecent 60-day HMM state counts:")
print(recent_60_states.value_counts().sort_index())

print("\nRecent 60-day HMM state proportions:")
print((recent_60_states.value_counts(normalize=True).sort_index() * 100).round(2))

print("\nMost recent state:", features_df["regime_hmm"].iloc[-1])
print("Most frequent state in last 60 days:", recent_60_states.mode()[0])


# --------------------------------------------------------------------------
# State duration analysis
# --------------------------------------------------------------------------
def calculate_state_durations(state_series: pd.Series) -> pd.DataFrame:
    """
    연속된 state 구간(run)의 길이를 계산한다.

    Parameters
    ----------
    state_series : pd.Series
        HMM/KMeans 등 regime label 시계열

    Returns
    -------
    pd.DataFrame
        각 run의 시작일, 종료일, state, 길이(duration)
    """
    runs = []
    current_state = state_series.iloc[0]
    start_idx = state_series.index[0]
    duration = 1

    for i in range(1, len(state_series)):
        if state_series.iloc[i] == current_state:
            duration += 1
        else:
            runs.append({
                "state": current_state,
                "start": start_idx,
                "end": state_series.index[i - 1],
                "duration": duration
            })
            current_state = state_series.iloc[i]
            start_idx = state_series.index[i]
            duration = 1

    # 마지막 run 추가
    runs.append({
        "state": current_state,
        "start": start_idx,
        "end": state_series.index[-1],
        "duration": duration
    })

    return pd.DataFrame(runs)


hmm_runs = calculate_state_durations(features_df["regime_hmm"])

print("\n" + "=" * 80)
print("HMM STATE DURATION SUMMARY")
print("=" * 80)

duration_summary = hmm_runs.groupby("state")["duration"].agg(
    ["count", "mean", "median", "max", "min"]
).round(2)

print(duration_summary)

print("\nOverall duration summary:")
print(hmm_runs["duration"].describe().round(2))
# ============================================================================
# BLOCK 11. REGIME VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

price_aligned = price_df.loc[features_df.index]

# KMeans
ax = axes[0]
for regime in sorted(features_df["regime_kmeans"].unique()):
    mask = features_df["regime_kmeans"] == regime
    ax.scatter(features_df.index[mask], price_aligned.loc[mask, "price"],
               label=f"Regime {regime}", alpha=0.6, s=20)
ax.plot(features_df.index, price_aligned["price"], "k-", alpha=0.2, linewidth=1)
ax.set_title("KMeans Regimes on Price")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True, alpha=0.3)

# HMM
ax = axes[1]
for state in sorted(features_df["regime_hmm"].unique()):
    mask = features_df["regime_hmm"] == state
    ax.scatter(features_df.index[mask], price_aligned.loc[mask, "price"],
               label=f"State {state}", alpha=0.6, s=20)
ax.plot(features_df.index, price_aligned["price"], "k-", alpha=0.2, linewidth=1)
ax.set_title("HMM Regimes on Price")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True, alpha=0.3)

# Recent 2 years
last_2y = features_df.tail(252 * 2)
ax = axes[2]
ax.fill_between(last_2y.index, 0, last_2y["regime_kmeans"], alpha=0.4, label="KMeans", step="mid")
ax.plot(last_2y.index, last_2y["regime_hmm"], "r-", marker="o", markersize=3, label="HMM", alpha=0.7)
ax.set_title("Recent Regime History (Last 2 Years)")
ax.set_ylabel("Regime ID")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# BLOCK 12. CURRENT REGIME DIAGNOSIS
# ============================================================================

recent_20 = features_df.tail(20)
recent_60 = features_df.tail(60)

print("=" * 80)
print("CURRENT REGIME STATUS")
print("=" * 80)

print("\n--- Last 20 Trading Days ---")
print("KMeans:")
print(recent_20["regime_kmeans"].value_counts().sort_index())
print("Most common KMeans regime:", recent_20["regime_kmeans"].mode()[0])

print("\nHMM:")
print(recent_20["regime_hmm"].value_counts().sort_index())
print("Most common HMM state:", recent_20["regime_hmm"].mode()[0])

print("\n--- Last 60 Trading Days ---")
print("KMeans:")
print(recent_60["regime_kmeans"].value_counts().sort_index())

print("\nHMM:")
print(recent_60["regime_hmm"].value_counts().sort_index())

print("\nLatest date:", features_df.index[-1].date())
print("Current KMeans regime:", features_df["regime_kmeans"].iloc[-1])
print("Current HMM state:", features_df["regime_hmm"].iloc[-1])

# ============================================================================
# BLOCK 13. REGIME MODEL COMPARISON
# ============================================================================

cm = confusion_matrix(features_df["regime_kmeans"], features_df["regime_hmm"])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=[f"HMM-{i}" for i in sorted(features_df["regime_hmm"].unique())],
    yticklabels=[f"KMeans-{i}" for i in sorted(features_df["regime_kmeans"].unique())]
)
ax.set_title("Confusion Matrix: KMeans vs HMM")
ax.set_xlabel("HMM")
ax.set_ylabel("KMeans")
plt.tight_layout()
plt.show()

print("Confusion Matrix:")
print(cm)
print(f"Diagonal agreement rate: {np.trace(cm) / np.sum(cm) * 100:.2f}%")

regime_corr = features_df[["regime_kmeans", "regime_hmm"]].corr().iloc[0, 1]
print("Correlation between regime IDs:", round(regime_corr, 4))

# ============================================================================
# BLOCK 14. FORWARD RETURN CALCULATION
# ============================================================================

def calculate_forward_returns(price_series, forward_periods=[63, 126]):
    """
    Calculate forward returns for specified horizons.
    """
    returns_dict = {}
    for period in forward_periods:
        forward_price = price_series.shift(-period)
        forward_ret = (forward_price / price_series) - 1
        returns_dict[f"{period}d_return"] = forward_ret
    return returns_dict

forward_returns = calculate_forward_returns(price_df["price"], forward_periods=[FWD_3M, FWD_6M])

for col, series in forward_returns.items():
    features_df[col] = series

features_df_with_forward = features_df.dropna()

print("✓ Forward returns added")
display(features_df_with_forward[[f"{FWD_3M}d_return", f"{FWD_6M}d_return"]].describe().round(4))

# ============================================================================
# BLOCK 15. REGIME CONDITIONAL FORWARD STATS
# ============================================================================

def summarize_forward_by_regime(df, regime_col, ret3_col="63d_return", ret6_col="126d_return"):
    output = {}

    for regime in sorted(df[regime_col].unique()):
        temp = df[df[regime_col] == regime]
        r3 = temp[ret3_col]
        r6 = temp[ret6_col]

        stats = {
            "count": len(temp),
            "3m_mean": r3.mean(),
            "3m_median": r3.median(),
            "3m_std": r3.std(),
            "3m_skew": skew(r3.dropna()),
            "3m_kurt": kurtosis(r3.dropna()),
            "6m_mean": r6.mean(),
            "6m_median": r6.median(),
            "6m_std": r6.std(),
            "6m_skew": skew(r6.dropna()),
            "6m_kurt": kurtosis(r6.dropna()),
            "3m_prob_loss15": (r3 < LOSS_THRESHOLD).mean(),
            "3m_prob_tail20": (abs(r3) > TAIL_THRESHOLD).mean(),
            "6m_prob_loss15": (r6 < LOSS_THRESHOLD).mean(),
            "6m_prob_tail20": (abs(r6) > TAIL_THRESHOLD).mean(),
        }
        output[regime] = stats

    return output

kmeans_forward_stats = summarize_forward_by_regime(features_df_with_forward, "regime_kmeans")
hmm_forward_stats = summarize_forward_by_regime(features_df_with_forward, "regime_hmm")

print("✓ Regime conditional forward stats computed")

# ============================================================================
# BLOCK 16. KMEANS FORWARD RETURN ANALYSIS
# ============================================================================

print("=" * 80)
print("FORWARD RETURNS BY KMEANS REGIME")
print("=" * 80)

for regime, stats in kmeans_forward_stats.items():
    print(f"\nKMeans Regime {regime} (n={stats['count']})")
    print(f"  3M Mean: {stats['3m_mean']*100:7.2f}%")
    print(f"  3M Std : {stats['3m_std']*100:7.2f}%")
    print(f"  3M P(Return < -15%): {stats['3m_prob_loss15']*100:7.2f}%")
    print(f"  3M P(|Return| > 20%): {stats['3m_prob_tail20']*100:7.2f}%")
    print(f"  6M Mean: {stats['6m_mean']*100:7.2f}%")
    print(f"  6M Std : {stats['6m_std']*100:7.2f}%")
    print(f"  6M P(Return < -15%): {stats['6m_prob_loss15']*100:7.2f}%")
    print(f"  6M P(|Return| > 20%): {stats['6m_prob_tail20']*100:7.2f}%")

# ============================================================================
# BLOCK 17. HMM FORWARD RETURN ANALYSIS
# ============================================================================

print("=" * 80)
print("FORWARD RETURNS BY HMM STATE")
print("=" * 80)

for state, stats in hmm_forward_stats.items():
    print(f"\nHMM State {state} (n={stats['count']})")
    print(f"  3M Mean: {stats['3m_mean']*100:7.2f}%")
    print(f"  3M Std : {stats['3m_std']*100:7.2f}%")
    print(f"  3M P(Return < -15%): {stats['3m_prob_loss15']*100:7.2f}%")
    print(f"  3M P(|Return| > 20%): {stats['3m_prob_tail20']*100:7.2f}%")
    print(f"  6M Mean: {stats['6m_mean']*100:7.2f}%")
    print(f"  6M Std : {stats['6m_std']*100:7.2f}%")
    print(f"  6M P(Return < -15%): {stats['6m_prob_loss15']*100:7.2f}%")
    print(f"  6M P(|Return| > 20%): {stats['6m_prob_tail20']*100:7.2f}%")

# ============================================================================
# BLOCK 17.5. HMM FORWARD RETURN ASYMMETRY CHECK
# ============================================================================
# 목적:
# 각 HMM state별로 forward return 분포의 상방/하방 비대칭성을 확인한다.

def summarize_asymmetry_by_state(df, regime_col="regime_hmm", ret_cols=["63d_return", "126d_return"]):
    results = []

    for state in sorted(df[regime_col].unique()):
        temp = df[df[regime_col] == state]

        row = {"state": state, "count": len(temp)}

        for col in ret_cols:
            r = temp[col].dropna()

            row[f"{col}_mean"] = r.mean()
            row[f"{col}_median"] = r.median()
            row[f"{col}_q10"] = r.quantile(0.10)
            row[f"{col}_q25"] = r.quantile(0.25)
            row[f"{col}_q75"] = r.quantile(0.75)
            row[f"{col}_q90"] = r.quantile(0.90)

            row[f"{col}_p_down_15"] = (r < -0.15).mean()
            row[f"{col}_p_up_15"] = (r > 0.15).mean()
            row[f"{col}_p_abs_20"] = (abs(r) > 0.20).mean()

        results.append(row)

    return pd.DataFrame(results)


hmm_asym_df = summarize_asymmetry_by_state(features_df_with_forward)

print("=" * 100)
print("HMM STATE FORWARD RETURN ASYMMETRY SUMMARY")
print("=" * 100)

display_cols = [
    "state", "count",
    "63d_return_mean", "63d_return_median", "63d_return_q10", "63d_return_q90",
    "63d_return_p_down_15", "63d_return_p_up_15", "63d_return_p_abs_20",
    "126d_return_mean", "126d_return_median", "126d_return_q10", "126d_return_q90",
    "126d_return_p_down_15", "126d_return_p_up_15", "126d_return_p_abs_20"
]

print(hmm_asym_df[display_cols].round(4).to_string(index=False))

# ============================================================================
# BLOCK 17.6. HMM FORWARD RETURN BOXPLOT
# ============================================================================
# 목적:
# 각 HMM state별 3M/6M forward return 분포를 boxplot으로 비교

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3M
sns.boxplot(
    data=features_df_with_forward,
    x="regime_hmm",
    y="63d_return",
    ax=axes[0]
)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].axhline(-0.15, color="red", linestyle="--", linewidth=1, label="-15%")
axes[0].axhline(0.15, color="blue", linestyle="--", linewidth=1, label="+15%")
axes[0].set_title("3-Month Forward Return by HMM State")
axes[0].set_xlabel("HMM State")
axes[0].set_ylabel("3M Forward Return")
axes[0].legend()

# 6M
sns.boxplot(
    data=features_df_with_forward,
    x="regime_hmm",
    y="126d_return",
    ax=axes[1]
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].axhline(-0.15, color="red", linestyle="--", linewidth=1, label="-15%")
axes[1].axhline(0.15, color="blue", linestyle="--", linewidth=1, label="+15%")
axes[1].set_title("6-Month Forward Return by HMM State")
axes[1].set_xlabel("HMM State")
axes[1].set_ylabel("6M Forward Return")
axes[1].legend()

plt.tight_layout()
plt.show()

# ============================================================================
# BLOCK 18. FORWARD RETURN DISTRIBUTION PLOTS
# ============================================================================

fig, axes = plt.subplots(HMM_STATES, 2, figsize=(16, 4 * HMM_STATES))
if HMM_STATES == 1:
    axes = [axes]

for idx, state in enumerate(sorted(features_df_with_forward["regime_hmm"].unique())):
    temp = features_df_with_forward[features_df_with_forward["regime_hmm"] == state]
    r3 = temp["63d_return"] * 100
    r6 = temp["126d_return"] * 100

    ax = axes[idx][0]
    ax.hist(r3, bins=50, alpha=0.7, color="mediumseagreen", edgecolor="black")
    ax.axvline(r3.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {r3.mean():.2f}%")
    ax.axvline(-15, color="orange", linestyle="--", linewidth=2, label="-15%")
    ax.axvline(20, color="blue", linestyle="--", linewidth=1, label="±20%")
    ax.axvline(-20, color="blue", linestyle="--", linewidth=1)
    ax.set_title(f"HMM State {state}: 3M Forward Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[idx][1]
    ax.hist(r6, bins=50, alpha=0.7, color="lightcoral", edgecolor="black")
    ax.axvline(r6.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {r6.mean():.2f}%")
    ax.axvline(-15, color="orange", linestyle="--", linewidth=2, label="-15%")
    ax.axvline(20, color="blue", linestyle="--", linewidth=1, label="±20%")
    ax.axvline(-20, color="blue", linestyle="--", linewidth=1)
    ax.set_title(f"HMM State {state}: 6M Forward Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# BLOCK 19. CURRENT REGIME INTERPRETATION
# ============================================================================

current_hmm_state = features_df["regime_hmm"].iloc[-1]
current_date = features_df.index[-1].date()

print("=" * 80)
print("CURRENT MARKET REGIME INTERPRETATION")
print("=" * 80)
print("Current date:", current_date)
print("Current HMM state:", current_hmm_state)

current_state_data = features_df_with_forward[
    features_df_with_forward["regime_hmm"] == current_hmm_state
]

r3 = current_state_data["63d_return"]
r6 = current_state_data["126d_return"]

print("\nCurrent-state conditional statistics:")
print(f"3M mean return:          {r3.mean()*100:7.2f}%")
print(f"3M std:                  {r3.std()*100:7.2f}%")
print(f"3M P(Return < -15%):     {(r3 < LOSS_THRESHOLD).mean()*100:7.2f}%")
print(f"3M P(|Return| > 20%):    {(abs(r3) > TAIL_THRESHOLD).mean()*100:7.2f}%")

print(f"\n6M mean return:          {r6.mean()*100:7.2f}%")
print(f"6M std:                  {r6.std()*100:7.2f}%")
print(f"6M P(Return < -15%):     {(r6 < LOSS_THRESHOLD).mean()*100:7.2f}%")
print(f"6M P(|Return| > 20%):    {(abs(r6) > TAIL_THRESHOLD).mean()*100:7.2f}%")

# ============================================================================
# BLOCK 20. V1 CONCLUSION
# ============================================================================

print("=" * 80)
print("PIPELINE V1 CONCLUSION")
print("=" * 80)

print("""
이 노트북의 v1 목적은 '현재 시장 레짐이 무엇이며,
그 레짐에서 향후 3~6개월 급락 또는 대변동 확률이 과거에 얼마나 높았는가'를 확인하는 것이다.

즉, 현재 시장이 구조화상품(예: ELD)이 유리한 국면인지 판단하기 위한
사전 진단 단계로 사용된다.
""")

# ============================================================================
# BLOCK 20.5. SAVE OUTPUTS FOR PIPELINE V2
# ============================================================================
# 목적:
# pipeline_v2에서 바로 불러올 수 있도록
# v1의 핵심 결과물을 csv 파일로 저장한다.

import os
import json

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. v2에서 핵심 입력으로 사용할 데이터
features_df_with_forward.to_csv(
    os.path.join(SAVE_DIR, "features_df_with_forward.csv"),
    encoding="utf-8-sig"
)

# 2. HMM state별 feature 평균
hmm_profile.to_csv(
    os.path.join(SAVE_DIR, "hmm_profile.csv"),
    encoding="utf-8-sig"
)

# 3. HMM transition matrix
transmat_df.to_csv(
    os.path.join(SAVE_DIR, "transmat_df.csv"),
    encoding="utf-8-sig"
)

print("✓ pipeline_v1 outputs saved successfully")
print(f"Saved files in: {SAVE_DIR}")
print("- features_df_with_forward.csv")
print("- hmm_profile.csv")
print("- transmat_df.csv")

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

current_regime_summary = {
    "latest_date": str(features_df.index[-1].date()),
    "current_kmeans_regime": int(features_df["regime_kmeans"].iloc[-1]),
    "current_hmm_state": int(features_df["regime_hmm"].iloc[-1]),
    "recent_20_kmeans_counts": {
        str(k): int(v) for k, v in features_df.tail(20)["regime_kmeans"].value_counts().sort_index().items()
    },
    "recent_20_hmm_counts": {
        str(k): int(v) for k, v in features_df.tail(20)["regime_hmm"].value_counts().sort_index().items()
    },
    "recent_60_kmeans_counts": {
        str(k): int(v) for k, v in features_df.tail(60)["regime_kmeans"].value_counts().sort_index().items()
    },
    "recent_60_hmm_counts": {
        str(k): int(v) for k, v in features_df.tail(60)["regime_hmm"].value_counts().sort_index().items()
    }
}

with open(os.path.join(SAVE_DIR, "current_regime_summary.json"), "w", encoding="utf-8") as f:
    json.dump(current_regime_summary, f, ensure_ascii=False, indent=2)

print("✓ current_regime_summary.json saved successfully")

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