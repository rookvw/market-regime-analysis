# ============================================================================
# PIPELINE V2 - BLOCK 1. IMPORTS & BASIC CONFIG
# ============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

print("✓ pipeline_v2 started")

# ============================================================================
# PIPELINE V2 - BLOCK 2. LOAD INPUT FROM PIPELINE V1 OUTPUT FILES
# ============================================================================

SAVE_DIR = "outputs"

# 1. main dataframe
features_df_with_forward = pd.read_csv(
    os.path.join(SAVE_DIR, "features_df_with_forward.csv"),
    index_col=0,
    parse_dates=True
)

# 2. hmm profile
hmm_profile = pd.read_csv(
    os.path.join(SAVE_DIR, "hmm_profile.csv"),
    index_col=0
)

# 3. hmm transition matrix
transmat_df = pd.read_csv(
    os.path.join(SAVE_DIR, "transmat_df.csv"),
    index_col=0
)

# 4. current regime summary
with open(os.path.join(SAVE_DIR, "current_regime_summary.json"), "r", encoding="utf-8") as f:
    current_regime_summary = json.load(f)

current_hmm_state = current_regime_summary["current_hmm_state"]
current_kmeans_regime = current_regime_summary["current_kmeans_regime"]

print("✓ v1 output loaded successfully")
print("Current HMM state:", current_hmm_state)
print("Current KMeans regime:", current_kmeans_regime)
print("Data shape:", features_df_with_forward.shape)

print("\nColumns:")
print(features_df_with_forward.columns.tolist())

# ============================================================================
# PIPELINE V2 - BLOCK 3. PRODUCT ASSUMPTIONS
# ============================================================================
# 목적:
# ETF, ELD, IBK 정기예금의 payoff 비교를 위한 기본 가정을 설정한다.
#
# 비교 horizon:
# - 현재 연구의 핵심은 "현재 레짐에서 향후 어떤 결과가 나타나는가"이므로
#   우선 6개월 forward return(126거래일 기준)을 메인으로 사용한다.
#
# 상품 정의:
# 1. ETF
#    - KOSPI 지수에 직접 노출되는 상품으로 가정
#    - payoff = 시장 수익률
#
# 2. IBK 정기예금
#    - 원금보장 + 확정금리
#    - payoff = 연이율 × 투자기간
#
# 3. ELD
#    - 원금보장 + 제한적 상방 참여
#    - payoff = min(max(alpha * R, 0), cap)

# -----------------------------
# Analysis horizon
# -----------------------------
MAIN_HORIZON_COL = "126d_return"   # 6개월 forward return
MAIN_HORIZON_YEARS = 0.5           # 6개월 = 0.5년

# -----------------------------
# Deposit assumption (IBK)
# -----------------------------
# 실제 IBK 공시금리를 확인한 뒤 여기에 입력
# 예: 연 3.10%면 0.031
DEPOSIT_ANNUAL_RATE = 0.03   # <-- 임시값, 나중에 실제 IBK 금리로 교체

# -----------------------------
# ELD assumption
# -----------------------------
ELD_PARTICIPATION = 1.0      # 상승분 100% 반영
ELD_CAP = 0.12               # 최대 수익률 12%
ELD_FLOOR = 0.0              # 원금보장: 하락 시 0%

# -----------------------------
# Optional sensitivity settings
# -----------------------------
ELD_SCENARIOS = {
    "base": {
        "participation": 1.0,
        "cap": 0.12
    },
    "conservative": {
        "participation": 0.8,
        "cap": 0.10
    },
    "aggressive": {
        "participation": 1.0,
        "cap": 0.15
    }
}

print("✓ Product assumptions configured")
print(f"Main horizon           : {MAIN_HORIZON_COL} ({MAIN_HORIZON_YEARS:.1f} year)")
print(f"Deposit annual rate    : {DEPOSIT_ANNUAL_RATE:.2%}")
print(f"ELD participation      : {ELD_PARTICIPATION:.2%}")
print(f"ELD cap                : {ELD_CAP:.2%}")
print(f"ELD floor              : {ELD_FLOOR:.2%}")

print("\nELD scenario set:")
for name, params in ELD_SCENARIOS.items():
    print(f"  {name:12s} | participation={params['participation']:.0%}, cap={params['cap']:.0%}")

# ============================================================================
# PIPELINE V2 - BLOCK 4. PAYOFF FUNCTIONS
# ============================================================================
# 목적:
# ETF, IBK 정기예금, ELD의 payoff를 함수로 정의한다.

def payoff_etf(market_return):
    """
    ETF payoff
    ----------------
    시장 수익률을 그대로 반영한다.

    Parameters
    ----------
    market_return : float or np.ndarray
        시장 수익률

    Returns
    -------
    float or np.ndarray
        ETF payoff
    """
    return market_return


def payoff_deposit(annual_rate=DEPOSIT_ANNUAL_RATE, horizon_years=MAIN_HORIZON_YEARS):
    """
    정기예금 payoff
    ----------------
    시장과 무관하게 확정금리를 지급한다.

    수식:
        payoff = annual_rate * horizon_years

    Parameters
    ----------
    annual_rate : float
        연이율
    horizon_years : float
        투자기간 (연 단위)

    Returns
    -------
    float
        예금 수익률
    """
    return annual_rate * horizon_years


def payoff_eld(market_return, participation=ELD_PARTICIPATION, cap=ELD_CAP, floor=ELD_FLOOR):
    """
    ELD payoff
    ----------------
    원금보장 + 제한적 상방참여 구조

    수식:
        payoff = min(max(participation * market_return, floor), cap)

    해석:
    - 시장 수익률이 음수면 floor(보통 0)
    - 시장 수익률이 양수면 participation만큼 반영
    - 단 cap 이상은 제한

    Parameters
    ----------
    market_return : float or np.ndarray
        시장 수익률
    participation : float
        상승분 참여율
    cap : float
        최대 수익률
    floor : float
        최소 수익률 (원금보장형이면 보통 0)

    Returns
    -------
    float or np.ndarray
        ELD payoff
    """
    return np.minimum(np.maximum(participation * market_return, floor), cap)

# ============================================================================
# PIPELINE V2 - BLOCK 4.5. PAYOFF FUNCTION CHECK
# ============================================================================

test_returns = np.array([-0.30, -0.15, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30])

print("=" * 80)
print("PAYOFF FUNCTION CHECK")
print("=" * 80)

print(f"{'Market Return':>15} | {'ETF':>10} | {'Deposit':>10} | {'ELD':>10}")
print("-" * 60)

for r in test_returns:
    etf_val = payoff_etf(r)
    dep_val = payoff_deposit()
    eld_val = payoff_eld(r)

    print(f"{r:15.2%} | {etf_val:10.2%} | {dep_val:10.2%} | {eld_val:10.2%}")

    # ============================================================================
# PIPELINE V2 - BLOCK 5. REGIME-WISE PRODUCT PAYOFF CALCULATION
# ============================================================================
# 목적:
# 각 HMM state에서 ETF / Deposit / ELD의 payoff 분포를 계산한다.

def build_product_payoff_table(df, regime_col="regime_hmm", return_col=MAIN_HORIZON_COL):
    """
    각 regime별로 ETF / Deposit / ELD payoff를 계산해 long-format 데이터프레임으로 반환한다.
    """
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

print("✓ Regime-wise product payoff table created")
print("Shape:", payoff_df.shape)
print("\nColumns:")
print(payoff_df.columns.tolist())

print("\nFirst 5 rows:")
print(payoff_df.head())

# ============================================================================
# PIPELINE V2 - BLOCK 5.5. PRODUCT PAYOFF SUMMARY BY HMM STATE
# ============================================================================
# 목적:
# 각 state별 ETF / Deposit / ELD의 payoff 요약통계를 계산한다.

def summarize_payoff_by_state(payoff_df):
    product_cols = ["ETF", "Deposit", "ELD"]
    results = []

    for state in sorted(payoff_df["state"].unique()):
        temp = payoff_df[payoff_df["state"] == state]

        for product in product_cols:
            r = temp[product]

            results.append({
                "state": state,
                "product": product,
                "count": len(r),
                "mean": r.mean(),
                "median": r.median(),
                "std": r.std(),
                "q10": r.quantile(0.10),
                "q90": r.quantile(0.90),
                "p_negative": (r < 0).mean(),
                "p_above_5pct": (r > 0.05).mean(),
                "max": r.max(),
                "min": r.min()
            })

    return pd.DataFrame(results)


payoff_summary_df = summarize_payoff_by_state(payoff_df)

print("✓ Product payoff summary created")
print("\nSummary preview:")
print(payoff_summary_df.round(4).head(12).to_string(index=False))

# ============================================================================
# PIPELINE V2 - BLOCK 6. MEAN PAYOFF BY STATE
# ============================================================================
# 목적:
# 각 HMM state에서 ETF / Deposit / ELD의 평균 payoff를 비교한다.

mean_payoff_table = payoff_summary_df.pivot(
    index="state",
    columns="product",
    values="mean"
)

print("Mean payoff table:")
print(mean_payoff_table.round(4))

mean_payoff_table.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Mean Payoff by HMM State")
plt.xlabel("HMM State")
plt.ylabel("Mean Payoff")
plt.xticks(rotation=0)
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 6.5. NEGATIVE PAYOFF PROBABILITY BY STATE
# ============================================================================
# 목적:
# 각 HMM state에서 상품별 손실확률(p_negative)을 비교한다.

negative_prob_table = payoff_summary_df.pivot(
    index="state",
    columns="product",
    values="p_negative"
)

print("Negative payoff probability table:")
print(negative_prob_table.round(4))

negative_prob_table.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.title("Negative Payoff Probability by HMM State")
plt.xlabel("HMM State")
plt.ylabel("Probability of Negative Payoff")
plt.xticks(rotation=0)
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 6.6. PAYOFF DISTRIBUTION BOXPLOT
# ============================================================================
# 목적:
# 각 state에서 ETF / Deposit / ELD payoff 분포를 boxplot으로 비교한다.

payoff_long_df = payoff_df.melt(
    id_vars=["state", "date", "market_return"],
    value_vars=["ETF", "Deposit", "ELD"],
    var_name="product",
    value_name="payoff"
)

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=payoff_long_df,
    x="state",
    y="payoff",
    hue="product"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Payoff Distribution by HMM State and Product")
plt.xlabel("HMM State")
plt.ylabel("Payoff")
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 6.7. CURRENT STATE PRODUCT COMPARISON
# ============================================================================
# 목적:
# 현재 HMM state에서 ETF / Deposit / ELD의 평균 payoff를 비교한다.

current_state_summary = payoff_summary_df[payoff_summary_df["state"] == current_hmm_state].copy()

print(f"Current state = {current_hmm_state}")
print(current_state_summary[["product", "mean", "median", "std", "p_negative", "q10", "q90"]].round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=current_state_summary,
    x="product",
    y="mean"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Mean Payoff in Current HMM State ({current_hmm_state})")
plt.ylabel("Mean Payoff")
plt.xlabel("Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 7. REGIME EVOLUTION SCENARIO SETUP
# ============================================================================
# 목적:
# 현재 HMM state를 기준으로 앞으로의 시장 전개를 두 가지 시나리오로 설정한다.
#
# Scenario A: State 3 persists
# Scenario B: State 3 transitions to State 2
'''
CURRENT_STATE = current_hmm_state
BULL_STATE = 3     # 현재 결과상 고변동 강세 상태
STRESS_STATE = 2   # 현재 결과상 고변동 하락 스트레스 상태

scenario_dict = {
    "state3_persist": BULL_STATE,
    "state3_to_state2": STRESS_STATE
}

print("✓ Regime evolution scenarios configured")
print("Current state:", CURRENT_STATE)
print("Scenario mapping:", scenario_dict)
'''

# 시나리오 구성:
# 1) 현재 state 지속 시나리오
# 2) 현재 state에서 stress state로 전이되는 시나리오
#
# 기본 설정:
# - stress_state는 경제적으로 의미 있는 위험 국면으로 직접 지정
# - 현재 연구에서는 state 2를 stress state로 사용
# ============================================================================

# ----------------------------------------------------------------------
# 1. 현재 state 확인
# ----------------------------------------------------------------------
current_state = current_hmm_state

# ----------------------------------------------------------------------
# 2. stress state 지정
# ----------------------------------------------------------------------
# 현재 해석 기준:
# State 2 = 고변동 스트레스 상태
stress_state = 2

# 만약 현재가 이미 stress_state라면, 보조 시나리오를 다른 state로 설정
if current_state == stress_state:
    # 예외 처리: 현재가 state 2라면 state 3을 보조 시나리오로 사용
    alt_state = 3 if current_state != 3 else 0
    scenario_map = {
        f"state{current_state}_persist": current_state,
        f"state{current_state}_to_state{alt_state}": alt_state
    }
else:
    scenario_map = {
        f"state{current_state}_persist": current_state,
        f"state{current_state}_to_state{stress_state}": stress_state
    }

print("✓ Regime evolution scenarios configured")
print("Current state:", current_state)
print("Scenario mapping:", scenario_map)


# ----------------------------------------------------------------------
# 3. 시나리오별 payoff 데이터 생성
# ----------------------------------------------------------------------
scenario_frames = []

for scenario_name, state_used in scenario_map.items():
    temp = payoff_df[payoff_df["state"] == state_used].copy()
    temp["scenario"] = scenario_name
    temp["state_used"] = state_used
    scenario_frames.append(temp)

scenario_payoff_df = pd.concat(scenario_frames, axis=0).reset_index(drop=True)

# 컬럼 순서 정리
scenario_payoff_df = scenario_payoff_df[
    ["scenario", "state_used", "date", "market_return", "ETF", "Deposit", "ELD"]
]

print("✓ Regime evolution scenario payoff table created")
print("Shape:", scenario_payoff_df.shape)
print(scenario_payoff_df.head())


# ----------------------------------------------------------------------
# 4. 시나리오 요약 테이블 생성
# ----------------------------------------------------------------------
scenario_summary_list = []

for scenario_name in scenario_payoff_df["scenario"].unique():
    temp = scenario_payoff_df[scenario_payoff_df["scenario"] == scenario_name]

    for product in ["ETF", "Deposit", "ELD"]:
        x = temp[product]

        scenario_summary_list.append({
            "scenario": scenario_name,
            "product": product,
            "count": len(x),
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "q10": x.quantile(0.10),
            "q90": x.quantile(0.90),
            "p_negative": (x < 0).mean(),
            "p_above_5pct": (x > 0.05).mean(),
            "max": x.max(),
            "min": x.min()
        })

scenario_summary_df = pd.DataFrame(scenario_summary_list)

print("✓ Scenario summary created")
print(scenario_summary_df.round(4).to_string(index=False))


# ----------------------------------------------------------------------
# 5. 시나리오별 평균 payoff / 음수 확률 테이블
# ----------------------------------------------------------------------
scenario_mean_table = scenario_summary_df.pivot(
    index="scenario",
    columns="product",
    values="mean"
).round(4)

scenario_negative_table = scenario_summary_df.pivot(
    index="scenario",
    columns="product",
    values="p_negative"
).round(4)

print("\nScenario mean payoff table:")
print(scenario_mean_table)

print("\nScenario negative payoff probability table:")
print(scenario_negative_table)
# ============================================================================
# PIPELINE V2 - BLOCK 7.5. BUILD REGIME EVOLUTION SCENARIO PAYOFFS
# ============================================================================
# 목적:
# 각 시나리오에 해당하는 HMM state의 historical forward return 분포를 사용해
# ETF / Deposit / ELD payoff를 계산한다.
'''
def build_regime_scenario_payoff_table(payoff_df, scenario_dict):
    records = []

    for scenario_name, state_value in scenario_dict.items():
        temp = payoff_df[payoff_df["state"] == state_value].copy()

        for _, row in temp.iterrows():
            records.append({
                "scenario": scenario_name,
                "state_used": state_value,
                "date": row["date"],
                "market_return": row["market_return"],
                "ETF": row["ETF"],
                "Deposit": row["Deposit"],
                "ELD": row["ELD"]
            })

    return pd.DataFrame(records)


scenario_payoff_df = build_regime_scenario_payoff_table(payoff_df, scenario_dict)

print("✓ Regime evolution scenario payoff table created")
print("Shape:", scenario_payoff_df.shape)
print(scenario_payoff_df.head())
'''
# ----------------------------------------------------------------------
# 1. Mean payoff by scenario
# ----------------------------------------------------------------------
scenario_mean_plot = scenario_summary_df.copy()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=scenario_mean_plot,
    x="scenario",
    y="mean",
    hue="product"
)
plt.title("Mean Payoff under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Mean Payoff")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# 2. Negative payoff probability by scenario
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.barplot(
    data=scenario_summary_df,
    x="scenario",
    y="p_negative",
    hue="product"
)
plt.title("Negative Payoff Probability under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Probability of Negative Payoff")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# 3. Payoff distribution by scenario and product
# ----------------------------------------------------------------------
scenario_long_df = scenario_payoff_df.melt(
    id_vars=["scenario", "state_used", "date", "market_return"],
    value_vars=["ETF", "Deposit", "ELD"],
    var_name="product",
    value_name="payoff"
)

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=scenario_long_df,
    x="scenario",
    y="payoff",
    hue="product"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Payoff Distribution under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Payoff")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 7.6. REGIME EVOLUTION SCENARIO SUMMARY
# ============================================================================
# 목적:
# 시나리오별 ETF / Deposit / ELD payoff 요약 통계를 계산한다.

def summarize_regime_scenarios(scenario_payoff_df):
    product_cols = ["ETF", "Deposit", "ELD"]
    results = []

    for scenario_name in sorted(scenario_payoff_df["scenario"].unique()):
        temp = scenario_payoff_df[scenario_payoff_df["scenario"] == scenario_name]

        for product in product_cols:
            r = temp[product]

            results.append({
                "scenario": scenario_name,
                "product": product,
                "count": len(r),
                "mean": r.mean(),
                "median": r.median(),
                "std": r.std(),
                "q10": r.quantile(0.10),
                "q90": r.quantile(0.90),
                "p_negative": (r < 0).mean(),
                "p_above_5pct": (r > 0.05).mean(),
                "max": r.max(),
                "min": r.min()
            })

    return pd.DataFrame(results)


scenario_summary_df = summarize_regime_scenarios(scenario_payoff_df)

print("✓ Scenario summary created")
print(scenario_summary_df.round(4).to_string(index=False))

# ============================================================================
# PIPELINE V2 - BLOCK 7.7. SCENARIO MEAN PAYOFF COMPARISON
# ============================================================================
# 목적:
# State 3 지속 vs State 2 전이 시나리오에서
# ETF / Deposit / ELD 평균 payoff를 비교한다.

scenario_mean_table = scenario_summary_df.pivot(
    index="scenario",
    columns="product",
    values="mean"
)

print("Scenario mean payoff table:")
print(scenario_mean_table.round(4))

scenario_mean_table.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Mean Payoff under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Mean Payoff")
plt.xticks(rotation=0)
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 7.8. SCENARIO NEGATIVE PAYOFF COMPARISON
# ============================================================================
# 목적:
# 각 시나리오에서 ETF / Deposit / ELD의 손실확률을 비교한다.

scenario_neg_table = scenario_summary_df.pivot(
    index="scenario",
    columns="product",
    values="p_negative"
)

print("Scenario negative payoff probability table:")
print(scenario_neg_table.round(4))

scenario_neg_table.plot(
    kind="bar",
    figsize=(12, 6)
)
plt.title("Negative Payoff Probability under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Probability of Negative Payoff")
plt.xticks(rotation=0)
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 7.9. SCENARIO PAYOFF DISTRIBUTION BOXPLOT
# ============================================================================
# 목적:
# 시나리오별 ETF / Deposit / ELD payoff 분포를 boxplot으로 비교한다.

scenario_long_df = scenario_payoff_df.melt(
    id_vars=["scenario", "state_used", "date", "market_return"],
    value_vars=["ETF", "Deposit", "ELD"],
    var_name="product",
    value_name="payoff"
)

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=scenario_long_df,
    x="scenario",
    y="payoff",
    hue="product"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Payoff Distribution under Regime Evolution Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Payoff")
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 8. TRANSITION-PROBABILITY-WEIGHTED SCENARIO
# ============================================================================
# 목적:
# 현재 HMM state에서 미래 state로의 전이확률을 반영하여,
# ETF / Deposit / ELD의 확률가중 기대 payoff를 계산한다.
'''
# 현재 state 이름 만들기
current_state_label = f"State_{current_hmm_state}"

# 현재 state에서 다음 state로 갈 전이확률
transition_probs = transmat_df.loc[current_state_label].copy()

print("✓ Transition probabilities from current state loaded")
print(f"Current HMM state: {current_hmm_state}")
print(transition_probs)
'''

# ----------------------------------------------------------------------
# 1. 현재 state의 전이확률 불러오기
# ----------------------------------------------------------------------
transition_probs = transmat_df.loc[f"State_{current_state}"]

print("✓ Transition probabilities from current state loaded")
print("Current HMM state:", current_state)
print(transition_probs)


# ----------------------------------------------------------------------
# 2. state별 평균 payoff 테이블
# ----------------------------------------------------------------------
state_mean_payoff = payoff_summary_df.pivot(
    index="state",
    columns="product",
    values="mean"
).round(4)

print("\nState-wise mean payoff table:")
print(state_mean_payoff)


# ----------------------------------------------------------------------
# 3. 전이확률 가중 기대 payoff 계산
# ----------------------------------------------------------------------
weighted_result = {}

for product in ["ETF", "Deposit", "ELD"]:
    weighted_payoff = 0.0

    for next_state in state_mean_payoff.index:
        prob = transition_probs[f"State_{next_state}"]
        state_mean = state_mean_payoff.loc[next_state, product]
        weighted_payoff += prob * state_mean

    weighted_result[product] = weighted_payoff

transition_weighted_df = pd.DataFrame.from_dict(
    weighted_result,
    orient="index",
    columns=["transition_weighted_mean_payoff"]
)

print("✓ Transition-weighted expected payoff calculated")
print(transition_weighted_df.round(4))


# ----------------------------------------------------------------------
# 4. 현재 state 평균과 비교
# ----------------------------------------------------------------------
current_state_mean = state_mean_payoff.loc[current_state].rename("current_state_mean")
comparison_df = pd.concat([current_state_mean, transition_weighted_df], axis=1)

print("\nComparison: Current-state mean vs Transition-weighted mean")
print(comparison_df.round(4))


# ----------------------------------------------------------------------
# 5. state contribution table
# ----------------------------------------------------------------------
contrib_rows = []

for next_state in state_mean_payoff.index:
    prob = transition_probs[f"State_{next_state}"]

    for product in ["ETF", "Deposit", "ELD"]:
        state_mean = state_mean_payoff.loc[next_state, product]
        contrib_rows.append({
            "current_state": current_state,
            "next_state": next_state,
            "transition_prob": prob,
            "product": product,
            "state_mean_payoff": state_mean,
            "weighted_contribution": prob * state_mean
        })

state_contribution_df = pd.DataFrame(contrib_rows)

print("\nState contribution table:")
print(state_contribution_df.round(4).to_string(index=False))

# ============================================================================
# PIPELINE V2 - BLOCK 8.5. STATE-WISE MEAN PAYOFF TABLE FOR WEIGHTING
# ============================================================================
# 목적:
# 각 HMM state별 상품 평균 payoff를 테이블로 정리한다.
'''
state_mean_payoff = payoff_summary_df.pivot(
    index="state",
    columns="product",
    values="mean"
).sort_index()

print("State-wise mean payoff table:")
print(state_mean_payoff.round(4))
'''
# ============================================================================
# BLOCK 8.5. VISUALIZATION: CURRENT-STATE VS TRANSITION-WEIGHTED PAYOFF
# ============================================================================

comparison_plot_df = comparison_df.reset_index().rename(columns={"index": "product"})
comparison_plot_df = comparison_plot_df.melt(
    id_vars="product",
    value_vars=["current_state_mean", "transition_weighted_mean_payoff"],
    var_name="scenario_type",
    value_name="mean_payoff"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=comparison_plot_df,
    x="product",
    y="mean_payoff",
    hue="scenario_type"
)
plt.title(f"Current-State vs Transition-Weighted Expected Payoff (State {current_state})")
plt.xlabel("Product")
plt.ylabel("Mean Payoff")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
# ============================================================================
# PIPELINE V2 - BLOCK 8.6. COMPUTE TRANSITION-WEIGHTED EXPECTED PAYOFF
# ============================================================================
# 목적:
# 현재 state의 전이확률을 이용해 상품별 확률가중 기대 payoff를 계산한다.

weighted_expected_payoff = {}

for product in ["ETF", "Deposit", "ELD"]:
    weighted_value = 0.0

    for next_state in state_mean_payoff.index:
        prob = transition_probs[f"State_{next_state}"]
        payoff_mean = state_mean_payoff.loc[next_state, product]
        weighted_value += prob * payoff_mean

    weighted_expected_payoff[product] = weighted_value

weighted_expected_payoff_df = pd.DataFrame.from_dict(
    weighted_expected_payoff,
    orient="index",
    columns=["transition_weighted_mean_payoff"]
)

print("✓ Transition-weighted expected payoff calculated")
print(weighted_expected_payoff_df.round(4))

# ============================================================================
# PIPELINE V2 - BLOCK 8.7. COMPARE CURRENT-STATE VS TRANSITION-WEIGHTED
# ============================================================================
# 목적:
# 현재 state 자체의 평균 payoff와 전이확률 가중 기대 payoff를 비교한다.

current_state_direct = payoff_summary_df[
    payoff_summary_df["state"] == current_hmm_state
][["product", "mean"]].copy()

current_state_direct = current_state_direct.rename(columns={"mean": "current_state_mean"})
current_state_direct = current_state_direct.set_index("product")

comparison_df = current_state_direct.join(weighted_expected_payoff_df)

print("Comparison: Current-state mean vs Transition-weighted mean")
print(comparison_df.round(4))

# ============================================================================
# PIPELINE V2 - BLOCK 8.8. VISUALIZE CURRENT VS TRANSITION-WEIGHTED PAYOFF
# ============================================================================
# 목적:
# 현재 state 직접 결과와 전이확률 가중 결과를 시각화한다.

comparison_plot_df = comparison_df.reset_index().melt(
    id_vars="product",
    value_vars=["current_state_mean", "transition_weighted_mean_payoff"],
    var_name="scenario_type",
    value_name="mean_payoff"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=comparison_plot_df,
    x="product",
    y="mean_payoff",
    hue="scenario_type"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Current-State vs Transition-Weighted Expected Payoff (State {current_hmm_state})")
plt.xlabel("Product")
plt.ylabel("Mean Payoff")
plt.tight_layout()
plt.show()

# ============================================================================
# PIPELINE V2 - BLOCK 8.9. STATE CONTRIBUTION TABLE
# ============================================================================
# 목적:
# 어떤 next state가 확률가중 기대 payoff에 얼마나 기여했는지 보여준다.

contribution_records = []

for next_state in state_mean_payoff.index:
    prob = transition_probs[f"State_{next_state}"]

    for product in ["ETF", "Deposit", "ELD"]:
        payoff_mean = state_mean_payoff.loc[next_state, product]
        contribution = prob * payoff_mean

        contribution_records.append({
            "current_state": current_hmm_state,
            "next_state": next_state,
            "transition_prob": prob,
            "product": product,
            "state_mean_payoff": payoff_mean,
            "weighted_contribution": contribution
        })

contribution_df = pd.DataFrame(contribution_records)

print("State contribution table:")
print(contribution_df.round(4).to_string(index=False))