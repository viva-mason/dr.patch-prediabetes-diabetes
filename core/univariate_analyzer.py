import numpy as np
import pandas as pd
import statsmodels.api as sm

class UnivariateLogisticAnalyzer:
    """단변량 로지스틱 회귀 분석을 수행하여 OR, 95% CI, p-value를 계산한다."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Args:
            X: 피처 행렬 (결측치가 처리되고 인코딩이 완료된 상태여야 함)
            y: 타깃 변수 (0 또는 1)
        """
        self.X = X
        self.y = y

    def analyze(self) -> pd.DataFrame:
        """각 피처에 대해 단변량 로지스틱 회귀를 수행하고 결과를 반환한다."""
        results = []

        for col in self.X.columns:
            x_col = self.X[col]
            
            # 결측치 제외 (이미 처리되어 있을 가능성이 높지만 안전을 위해)
            mask = x_col.notna() & self.y.notna()
            x_clean = x_col.loc[mask]
            y_clean = self.y.loc[mask]

            if len(x_clean) == 0 or y_clean.nunique() < 2:
                continue

            # 상수항 추가
            X_sm = sm.add_constant(x_clean)
            
            try:
                # 로지스틱 회귀 적합
                model = sm.Logit(y_clean, X_sm)
                fit = model.fit(disp=False)

                # 계수, p-value, 신뢰구간 추출
                coef = fit.params[col]
                p_val = fit.pvalues[col]
                conf_int = fit.conf_int().loc[col]

                # Odds Ratio 및 95% CI 계산
                or_val = np.exp(coef)
                ci_lower = np.exp(conf_int[0])
                ci_upper = np.exp(conf_int[1])

                results.append({
                    "Feature": col,
                    "OR": round(float(or_val), 3),
                    "95% CI Lower": round(float(ci_lower), 3),
                    "95% CI Upper": round(float(ci_upper), 3),
                    "p-value": round(float(p_val), 4),
                    "Significance": bool(p_val < 0.05)
                })
            except Exception:
                # 수렴 실패(완전 분리 등) 시 건너뜀
                pass

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values("OR", ascending=False).reset_index(drop=True)
            
        return res_df
