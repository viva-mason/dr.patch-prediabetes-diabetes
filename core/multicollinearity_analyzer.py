from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core import set_korean_font

set_korean_font()


class MulticollinearityAnalyzer:
    """전처리된 데이터셋의 다중공선성(VIF·상관계수)을 분석한다."""

    DEFAULT_EXCLUDE_COLUMNS: frozenset[str] = frozenset({
        "user_key",
        "current_checkup_date",
        "future_checkup_date",
        "selected_transition",
        "full_transition",
        "interval_days",
        "checkup_date",
    })

    REDUNDANT_OBJECT_COLUMNS: frozenset[str] = frozenset({"birthday"})

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        exclude_columns: Iterable[str] | None = None,
        vif_threshold: float = 10.0,
        correlation_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            df: 분석 대상 DataFrame
            label_col: 타깃 컬럼명
            exclude_columns: 피처에서 제외할 메타데이터 컬럼
            vif_threshold: 고다중공선성 판정 VIF 임계값
            correlation_threshold: 고상관 판정 |r| 임계값
        """
        if label_col not in df.columns:
            raise ValueError(f"label 컬럼 '{label_col}'을 찾을 수 없습니다.")

        self._df = df.copy()
        self.label_col = label_col
        self.exclude_columns = (
            frozenset(exclude_columns)
            if exclude_columns is not None
            else self.DEFAULT_EXCLUDE_COLUMNS
        )
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold

        self._y = self._df[self.label_col]
        self._feature_df, self._encoding_log = self._prepare_features()
        self._vif_table: pd.DataFrame | None = None
        self._correlation_matrix: pd.DataFrame | None = None

    @property
    def feature_count(self) -> int:
        """원-핫 인코딩 후 피처 수를 반환한다."""
        return self._feature_df.shape[1]

    @property
    def sample_count(self) -> int:
        """분석에 사용된 표본 수를 반환한다."""
        return self._feature_df.shape[0]

    def encoding_log(self) -> pd.DataFrame:
        """원-핫 인코딩·제외 컬럼 처리 내역을 반환한다."""
        return self._encoding_log.copy()

    def _one_hot_encode_column(
        self,
        series: pd.Series,
        column: str,
    ) -> tuple[pd.DataFrame, str]:
        """범주형 컬럼을 원-핫 인코딩한다. 유목이 2개이면 더미 1개만 유지한다."""
        category_count = int(series.nunique(dropna=True))
        if category_count <= 1:
            return pd.DataFrame(index=series.index), "constant"

        if category_count == 2:
            encoded = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            detail = "1 dummy column (binary, drop_first)"
        else:
            filled = series.fillna("missing")
            encoded = pd.get_dummies(filled, prefix=column, drop_first=True, dtype=float)
            detail = f"{encoded.shape[1]} dummy columns (drop_first)"

        return encoded, detail

    def _prepare_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """메타데이터를 제외하고 범주형은 원-핫 인코딩한 피처 행렬을 만든다."""
        drop_columns = set(self.exclude_columns) | {self.label_col}
        working = self._df.drop(columns=[c for c in drop_columns if c in self._df.columns])

        log_rows: list[dict[str, object]] = []
        for col in sorted(self.exclude_columns):
            if col in self._df.columns:
                log_rows.append({
                    "column": col,
                    "action": "excluded",
                    "detail": "metadata",
                })

        redundant_cols = [
            col for col in working.columns if col in self.REDUNDANT_OBJECT_COLUMNS
        ]
        if redundant_cols:
            working = working.drop(columns=redundant_cols)
            for col in redundant_cols:
                log_rows.append({
                    "column": col,
                    "action": "excluded",
                    "detail": "redundant_with_나이",
                })

        object_cols = working.select_dtypes(include=["object", "string"]).columns.tolist()
        numeric_cols = [col for col in working.columns if col not in object_cols]

        encoded_parts: list[pd.DataFrame] = []
        for col in object_cols:
            encoded, detail = self._one_hot_encode_column(working[col], col)
            if detail == "constant":
                continue
            encoded_parts.append(encoded)
            log_rows.append({
                "column": col,
                "action": "one_hot_encoded",
                "detail": detail,
            })

        numeric_part = working[numeric_cols].apply(pd.to_numeric, errors="coerce")
        encoded = (
            pd.concat(encoded_parts, axis=1)
            if encoded_parts
            else pd.DataFrame(index=working.index)
        )
        features = pd.concat([numeric_part, encoded], axis=1)
        features = features.loc[:, features.nunique(dropna=False) > 1]
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

        return features, pd.DataFrame(log_rows)

    def compute_vif(self) -> pd.DataFrame:
        """각 피처의 VIF(Variance Inflation Factor)를 계산한다."""
        x_values = self._feature_df.astype(float).values
        vif_rows: list[dict[str, float | str]] = []

        for idx, column in enumerate(self._feature_df.columns):
            vif_value = variance_inflation_factor(x_values, idx)
            if np.isinf(vif_value):
                vif_value = np.nan
            vif_rows.append({
                "feature": column,
                "vif": round(float(vif_value), 3) if pd.notna(vif_value) else np.nan,
                "high_vif": bool(pd.notna(vif_value) and vif_value >= self.vif_threshold),
            })

        self._vif_table = (
            pd.DataFrame(vif_rows)
            .sort_values("vif", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        return self._vif_table.copy()

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """피처 간 Pearson 상관계수 행렬을 계산한다."""
        self._correlation_matrix = self._feature_df.corr(method="pearson")
        return self._correlation_matrix.copy()

    def high_correlation_pairs(self) -> pd.DataFrame:
        """|r|이 임계값 이상인 상관쌍을 반환한다."""
        corr = self.compute_correlation_matrix()
        pairs: list[dict[str, object]] = []

        for row_idx, row_name in enumerate(corr.columns):
            for col_idx in range(row_idx + 1, len(corr.columns)):
                col_name = corr.columns[col_idx]
                value = corr.iloc[row_idx, col_idx]
                if pd.isna(value) or abs(value) < self.correlation_threshold:
                    continue
                pairs.append({
                    "feature_a": row_name,
                    "feature_b": col_name,
                    "correlation": round(float(value), 3),
                })

        result = pd.DataFrame(pairs)
        if result.empty:
            return result
        return result.sort_values(
            "correlation",
            key=lambda s: s.abs(),
            ascending=False,
        ).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """다중공선성 분석 요약 통계를 반환한다."""
        vif = self.compute_vif()
        high_vif = vif.loc[vif["high_vif"]]
        high_corr = self.high_correlation_pairs()

        rows = [{
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "vif_threshold": self.vif_threshold,
            "high_vif_count": int(high_vif.shape[0]),
            "max_vif": vif["vif"].max(skipna=True),
            "correlation_threshold": self.correlation_threshold,
            "high_correlation_pair_count": int(high_corr.shape[0]),
        }]
        return pd.DataFrame(rows)

    def plot_vif(
        self,
        output_path: Path | None = None,
        top_n: int = 20,
        show: bool = True,
    ) -> None:
        """VIF 상위 피처를 막대그래프로 시각화한다."""
        vif = self.compute_vif().dropna(subset=["vif"]).head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        sns.barplot(
            data=vif,
            y="feature",
            x="vif",
            hue="feature",
            palette="Reds_r",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.axvline(
            self.vif_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"VIF={self.vif_threshold:g}",
        )
        ax.set_title(f"VIF Top {top_n}")
        ax.set_xlabel("VIF")
        ax.set_ylabel("feature")
        ax.legend(loc="lower right")

        plt.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_correlation_heatmap(
        self,
        output_path: Path | None = None,
        top_n: int = 25,
        show: bool = True,
    ) -> None:
        """절대 상관계수가 큰 피처 위주로 히트맵을 그린다."""
        corr = self.compute_correlation_matrix()
        mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
        selected = mean_abs_corr.head(top_n).index.tolist()
        subset = corr.loc[selected, selected]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            subset,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.2,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"상관계수 히트맵 (Top {top_n})")

        plt.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def iterative_vif_elimination(
        self,
        threshold: float | None = None,
        protected_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """VIF ≥ threshold인 피처를 가장 높은 것부터 하나씩 반복 제거한다.

        보호 컬럼은 VIF가 높아도 제거하지 않는다.
        결과 DataFrame에 제거 순서·제거 시점 VIF·잔류 여부가 기록된다.

        Args:
            threshold: VIF 기준값. None이면 self.vif_threshold를 사용한다.
            protected_columns: 제거 대상에서 제외할 컬럼 이름 목록.

        Returns:
            columns: 제거된 컬럼, step: 제거 순서, vif_at_removal: 제거 시점 VIF
        """
        cutoff = threshold if threshold is not None else self.vif_threshold
        protected = frozenset(protected_columns) if protected_columns else frozenset()

        remaining = self._feature_df.copy()
        elimination_log: list[dict[str, object]] = []
        step = 0

        while True:
            x = remaining.astype(float).values
            vifs = {
                col: variance_inflation_factor(x, idx)
                for idx, col in enumerate(remaining.columns)
            }
            max_col = max(vifs, key=lambda c: vifs[c] if not np.isinf(vifs[c]) else np.inf)
            max_vif = vifs[max_col]

            if np.isinf(max_vif) or max_vif >= cutoff:
                candidates = {
                    col: v
                    for col, v in vifs.items()
                    if (np.isinf(v) or v >= cutoff) and col not in protected
                }
                if not candidates:
                    break
                drop_col = max(candidates, key=lambda c: candidates[c] if not np.isinf(candidates[c]) else np.inf)
                step += 1
                elimination_log.append({
                    "step": step,
                    "dropped_feature": drop_col,
                    "vif_at_removal": round(float(vifs[drop_col]), 3) if not np.isinf(vifs[drop_col]) else np.inf,
                    "protected": drop_col in protected,
                })
                remaining = remaining.drop(columns=[drop_col])
            else:
                break

        surviving_vifs = {
            col: round(float(variance_inflation_factor(remaining.astype(float).values, idx)), 3)
            for idx, col in enumerate(remaining.columns)
        }
        surviving_df = pd.DataFrame([
            {"survived_feature": col, "final_vif": vif}
            for col, vif in sorted(surviving_vifs.items(), key=lambda x: -x[1])
        ])

        self._elimination_log = pd.DataFrame(elimination_log)
        self._surviving_features = list(remaining.columns)
        self._surviving_vifs = surviving_df

        return self._elimination_log.copy()

    def elimination_result(self) -> dict[str, pd.DataFrame | list[str]]:
        """iterative_vif_elimination() 수행 후 생존 피처·VIF를 반환한다.

        Returns:
            "dropped": 제거 로그 DataFrame
            "survived": 생존 피처 VIF DataFrame
            "survived_columns": 생존 피처 이름 목록
        """
        if not hasattr(self, "_elimination_log"):
            raise RuntimeError("먼저 iterative_vif_elimination()을 호출하세요.")
        return {
            "dropped": self._elimination_log.copy(),
            "survived": self._surviving_vifs.copy(),
            "survived_columns": list(self._surviving_features),
        }

    def save_tables(self, output_dir: Path) -> None:
        """VIF·상관쌍·요약·인코딩 로그를 CSV로 저장한다."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.compute_vif().to_csv(output_dir / "vif.csv", index=False)
        self.high_correlation_pairs().to_csv(
            output_dir / "high_correlation_pairs.csv",
            index=False,
        )
        self.summary().to_csv(output_dir / "summary.csv", index=False)
        self.encoding_log().to_csv(output_dir / "encoding_log.csv", index=False)
        if hasattr(self, "_elimination_log"):
            self._elimination_log.to_csv(output_dir / "vif_elimination_log.csv", index=False)
            self._surviving_vifs.to_csv(output_dir / "vif_survived.csv", index=False)
