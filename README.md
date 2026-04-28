# 시장 레짐 분석 — KMeans · HMM 기반 ELD 합리성 평가

KOSPI-VKOSPI 데이터로 시장 국면(레짐)을 식별하고, 레짐별 ETF·정기예금·ELD의 기대 payoff를 정량 비교해 **ELD 투자 합리성을 손실회피 효용함수로 수치 검증**한 프로젝트.

## 결과

### Walk-forward 검증 — AUC 분포
![AUC Boxplot](outputs/figures/fig_v3_auc_boxplot.png)

### Out-of-Sample 레짐 전환 시각화
![Regime Transitions](outputs/figures/fig_v3_oos_regime_transitions.png)

---

## 핵심 결과

| 지표 | KMeans | HMM |
|---|---|---|
| AUC | 0.363 | **0.409** |
| Brier Score | - | 낮을수록 우수 |

- 현재 레짐 기준: ETF payoff **11.47%** vs ELD **7.26%** vs 예금 **1.50%**
- **λ ≥ 0.145** 이면 손실회피 투자자에게 ELD가 ETF보다 합리적임을 수치로 증명

## 방법론

| 단계 | 내용 |
|---|---|
| 데이터 | 2000–2026 KOSPI·VKOSPI 일별 |
| 변수 | 1M·3M 수익률, 실현변동성, MDD, MA이격도, VKOSPI z-score (7개) |
| 모델 | KMeans · HMM 4-state |
| 검증 | Walk-forward (AUC, Brier Score, Log Loss) |
| 효용함수 | U = mean − λ × loss_prob (손실회피 모델) |

## 기술 스택

`Python` `hmmlearn` `scikit-learn` `pandas` `numpy` `matplotlib`

## 실행

```bash
pip install hmmlearn scikit-learn pandas numpy matplotlib jupyter
jupyter notebook korean_market_regime_analysis.ipynb
```
