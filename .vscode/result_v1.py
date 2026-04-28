# ============================================================================
# RESULT_V1 - BLOCK 0. LOAD INPUTS
# ============================================================================
# 목적:
# result_v1을 standalone script로 실행할 수 있도록
# pipeline_v1 / v2 결과를 파일에서 불러온다.

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = "outputs"

# --------------------------------------------------
# 1. load v1 outputs
# --------------------------------------------------
features_df_with_forward = pd.read_csv(
    os.path.join(OUTPUT_DIR, "features_df_with_forward.csv"),
    index_col=0,
    parse_dates=True
)

with open(os.path.join(OUTPUT_DIR, "current_regime_summary.json"), "r", encoding="utf-8") as f:
    current_regime_summary = json.load(f)

current_hmm_state = current_regime_summary["current_hmm_state"]
current_kmeans_regime = current_regime_summary["current_kmeans_regime"]

print("✓ Inputs loaded successfully")
print("Current HMM state:", current_hmm_state)
print("Current KMeans regime:", current_kmeans_regime)
print("Shape:", features_df_with_forward.shape)
print(features_df_with_forward.head())

# ============================================================================
# RESULT_V1 - BLOCK 0.5. PRODUCT ASSUMPTIONS
# ============================================================================

MAIN_HORIZON_COL = "126d_return"
MAIN_HORIZON_YEARS = 0.5

DEPOSIT_ANNUAL_RATE = 0.03
ELD_PARTICIPATION = 1.0
ELD_CAP = 0.12
ELD_FLOOR = 0.0

print("✓ Product assumptions ready")
print("Main horizon:", MAIN_HORIZON_COL)
print("Deposit annual rate:", DEPOSIT_ANNUAL_RATE)
print("ELD participation:", ELD_PARTICIPATION)
print("ELD cap:", ELD_CAP)
print("ELD floor:", ELD_FLOOR)

# ============================================================================
# RESULT_V1 - BLOCK 0.6. PAYOFF FUNCTIONS
# ============================================================================

def payoff_etf(market_return):
    return market_return

def payoff_deposit(annual_rate=DEPOSIT_ANNUAL_RATE, horizon_years=MAIN_HORIZON_YEARS):
    return annual_rate * horizon_years

def payoff_eld(market_return, participation=ELD_PARTICIPATION, cap=ELD_CAP, floor=ELD_FLOOR):
    return np.minimum(np.maximum(participation * market_return, floor), cap)

# ============================================================================
# RESULT_V1 - BLOCK 0.7. BUILD PAYOFF DF
# ============================================================================
# 목적:
# current state 평가에 필요한 payoff_df 구성

def build_product_payoff_table(df, regime_col="regime_hmm", return_col=MAIN_HORIZON_COL):
    records = []

    for state in sorted(df[regime_col].unique()):
        temp = df[df[regime_col] == state].copy()
        market_r = temp[return_col].values

        etf_payoff = payoff_etf(market_r)
        deposit_payoff = np.repeat(payoff_deposit(), len(temp))
        eld_payoff = payoff_eld(market_r)

        for i in range(len(temp)):
            records.append({
                "state": state,
                "date": temp.index[i],
                "market_return": market_r[i],
                "ETF": etf_payoff[i],
                "Deposit": deposit_payoff[i],
                "ELD": eld_payoff[i]
            })

    return pd.DataFrame(records)

payoff_df = build_product_payoff_table(features_df_with_forward)

print("✓ payoff_df created")
print(payoff_df.head())

# ============================================================================
# RESULT_V1 - BLOCK 1. EXTRACT CURRENT-STATE PAYOFF DATA
# ============================================================================
# 목적:
# 현재 HMM state에서 ETF / Deposit / ELD payoff 데이터만 추출한다.

current_state = current_hmm_state

current_payoff_df = payoff_df[payoff_df["state"] == current_state].copy()

print("✓ Current-state payoff data extracted")
print("Current HMM state:", current_state)
print("Shape:", current_payoff_df.shape)
print(current_payoff_df.head())

# ============================================================================
# RESULT_V1 - BLOCK 2. DEFINE EVALUATION METRICS
# ============================================================================
# 목적:
# 현재 state에서 상품별 payoff 평가 지표를 계산하는 함수 정의

def loss_probability(x):
    x = np.asarray(x)
    return np.mean(x < 0)

def downside_mean(x):
    x = np.asarray(x)
    neg = x[x < 0]
    if len(neg) == 0:
        return 0.0
    return neg.mean()

def downside_semivariance(x):
    x = np.asarray(x)
    downside = np.minimum(x, 0.0)
    return np.mean(downside ** 2)

def utility_score_lossprob(x, lam=0.10):
    """
    Utility-like score:
    mean payoff - lam * loss probability
    """
    x = np.asarray(x)
    return x.mean() - lam * np.mean(x < 0)

def utility_score_downside(x, lam=1.0):
    """
    Utility-like score:
    mean payoff - lam * downside semivariance
    """
    x = np.asarray(x)
    return x.mean() - lam * downside_semivariance(x)

# ============================================================================
# RESULT_V1 - BLOCK 3. BUILD CURRENT-STATE METRIC TABLE
# ============================================================================
# 목적:
# 현재 state에서 ETF / Deposit / ELD의 핵심 평가 지표를 계산한다.

product_cols = ["ETF", "Deposit", "ELD"]

metric_rows = []

for product in product_cols:
    x = current_payoff_df[product].values

    metric_rows.append({
        "product": product,
        "count": len(x),
        "mean_payoff": np.mean(x),
        "median_payoff": np.median(x),
        "std_payoff": np.std(x, ddof=1),
        "q10": np.quantile(x, 0.10),
        "q90": np.quantile(x, 0.90),
        "loss_prob": loss_probability(x),
        "downside_mean": downside_mean(x),
        "downside_semivariance": downside_semivariance(x)
    })

current_metric_df = pd.DataFrame(metric_rows)

print("✓ Current-state metric table created")
print(current_metric_df.round(4).to_string(index=False))

# ============================================================================
# RESULT_V1 - BLOCK 4. UTILITY SCORE TABLE
# ============================================================================
# 목적:
# lambda 값에 따라 상품별 utility score를 계산한다.

lambda_grid_lossprob = [0.00, 0.05, 0.10, 0.20]
lambda_grid_downside = [0.00, 1.00, 3.00, 5.00]

utility_rows = []

for product in product_cols:
    x = current_payoff_df[product].values

    for lam in lambda_grid_lossprob:
        utility_rows.append({
            "product": product,
            "score_type": "mean_minus_lambda_lossprob",
            "lambda": lam,
            "score": utility_score_lossprob(x, lam=lam)
        })

    for lam in lambda_grid_downside:
        utility_rows.append({
            "product": product,
            "score_type": "mean_minus_lambda_downsideSemiVar",
            "lambda": lam,
            "score": utility_score_downside(x, lam=lam)
        })

utility_df = pd.DataFrame(utility_rows)

print("✓ Utility score table created")
print(utility_df.round(6).head(20).to_string(index=False))

# ============================================================================
# RESULT_V1 - BLOCK 5. BEST PRODUCT BY CRITERION
# ============================================================================
# 목적:
# 각 lambda와 score_type에서 최고 점수 상품을 선택한다.

best_product_df = (
    utility_df.sort_values(["score_type", "lambda", "score"], ascending=[True, True, False])
              .groupby(["score_type", "lambda"], as_index=False)
              .first()
)

print("✓ Best product by criterion selected")
print(best_product_df.round(6).to_string(index=False))

# ============================================================================
# RESULT_V1 - BLOCK 6. VISUALIZE UTILITY CURVES
# ============================================================================
# 목적:
# lambda 변화에 따라 어떤 상품이 우세한지 시각화한다.

# 1) mean - lambda * loss probability
plot_df1 = utility_df[utility_df["score_type"] == "mean_minus_lambda_lossprob"].copy()

plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df1, x="lambda", y="score", hue="product", marker="o")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Utility Score in Current State {current_state}\n(mean - λ × loss probability)")
plt.xlabel("Risk Aversion λ")
plt.ylabel("Utility Score")
plt.tight_layout()
plt.show()

# 2) mean - lambda * downside semivariance
plot_df2 = utility_df[utility_df["score_type"] == "mean_minus_lambda_downsideSemiVar"].copy()

plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_df2, x="lambda", y="score", hue="product", marker="o")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Utility Score in Current State {current_state}\n(mean - λ × downside semivariance)")
plt.xlabel("Risk Aversion λ")
plt.ylabel("Utility Score")
plt.tight_layout()
plt.show()

# ============================================================================
# RESULT_V1 - BLOCK 7. DECISION SUMMARY
# ============================================================================
# 목적:
# 보고서에 넣기 쉽게 현재 state에서의 해석 요약표를 만든다.

decision_summary = current_metric_df.copy()

decision_summary["rank_mean_payoff"] = decision_summary["mean_payoff"].rank(ascending=False, method="min")
decision_summary["rank_loss_prob"] = decision_summary["loss_prob"].rank(ascending=True, method="min")
decision_summary["rank_q10"] = decision_summary["q10"].rank(ascending=False, method="min")

print("✓ Decision summary created")
print(decision_summary.round(4).to_string(index=False))

# ============================================================================
# RESULT_V1 - BLOCK 8. SAVE OUTPUTS
# ============================================================================
# 목적:
# result_v1 결과를 저장한다.

os.makedirs("outputs/tables", exist_ok=True)

current_metric_df.to_csv("outputs/tables/result_v1_current_metric_df.csv", index=False)
utility_df.to_csv("outputs/tables/result_v1_utility_df.csv", index=False)
best_product_df.to_csv("outputs/tables/result_v1_best_product_df.csv", index=False)
decision_summary.to_csv("outputs/tables/result_v1_decision_summary.csv", index=False)

print("✓ result_v1 outputs saved")
print("- outputs/tables/result_v1_current_metric_df.csv")
print("- outputs/tables/result_v1_utility_df.csv")
print("- outputs/tables/result_v1_best_product_df.csv")
print("- outputs/tables/result_v1_decision_summary.csv")