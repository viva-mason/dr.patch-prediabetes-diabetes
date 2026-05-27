from __future__ import annotations

import pandas as pd


class MissingRateColumnFilter:
    """결측 비율이 임계값 이상인 검진 인자(컬럼)를 제거한다."""

    DEFAULT_PRESERVE_COLUMNS: frozenset[str] = frozenset({
        "user_key",
        "current_checkup_date",
        "future_checkup_date",
        "label",
        "selected_transition",
        "full_transition",
        "interval_days",
        "checkup_date",
        "gender",
        "birthday",
        "나이",
    })

    def __init__(
        self,
        max_missing_rate: float = 0.20,
        preserve_columns: frozenset[str] | None = None,
        always_drop_columns: frozenset[str] | None = None,
    ) -> None:
        """
        Args:
            max_missing_rate: 이 비율 이상 결측이면 제거 (0.20 = 20%)
            preserve_columns: 결측률과 관계없이 유지할 컬럼명
            always_drop_columns: 결측률과 관계없이 항상 제거할 컬럼명
        """
        if not 0 <= max_missing_rate <= 1:
            raise ValueError("max_missing_rate는 0~1 사이여야 합니다.")
        self.max_missing_rate = max_missing_rate
        self.preserve_columns = (
            preserve_columns
            if preserve_columns is not None
            else self.DEFAULT_PRESERVE_COLUMNS
        )
        self.always_drop_columns = (
            always_drop_columns if always_drop_columns is not None else frozenset()
        )

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """결측률이 임계값 이상인 컬럼을 제거한 DataFrame과 제거 내역을 반환한다."""
        n = len(df)
        rows: list[dict[str, object]] = []
        drop_columns: list[str] = []

        for col in df.columns:
            if col in self.preserve_columns:
                continue

            missing_count = int(df[col].isna().sum())
            missing_pct = round(missing_count / n * 100, 2) if n else 0.0
            force_dropped = col in self.always_drop_columns
            rate_dropped = missing_pct >= self.max_missing_rate * 100
            dropped = force_dropped or rate_dropped

            rows.append(
                {
                    "field": col,
                    "missing_count": missing_count,
                    "missing_pct": missing_pct,
                    "dropped": dropped,
                    "drop_reason": (
                        "always_drop"
                        if force_dropped
                        else ("missing_rate" if rate_dropped else "")
                    ),
                }
            )
            if dropped:
                drop_columns.append(col)

        report = pd.DataFrame(rows)
        if not report.empty:
            report = report.sort_values(
                ["dropped", "missing_pct"],
                ascending=[False, False],
            ).reset_index(drop=True)

        kept_columns = [c for c in df.columns if c not in drop_columns]
        return df[kept_columns].copy(), report

    def dropped_summary(self, report: pd.DataFrame) -> pd.DataFrame:
        """제거된 컬럼만 반환한다."""
        if report.empty:
            return pd.DataFrame(columns=["field", "missing_count", "missing_pct", "dropped"])
        return report.loc[report["dropped"]].reset_index(drop=True)
