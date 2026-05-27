from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.qualitative_lab_encoder import QualitativeLabEncoder


@dataclass
class NumericCoercionRecord:
    """대부분 수치인 컬럼의 텍스트 공백·numeric 변환 요약."""

    field: str
    numeric_ratio: float
    blanked_count: int
    excluded_values: str
    dtype_before: str
    dtype_after: str


class MostlyNumericCoercer:
    """비결측 값의 대부분이 수치로 해석되는 컬럼에서 텍스트만 공백 처리하고 numeric으로 변환한다."""

    DEFAULT_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
        {
            "user_key",
            "current_checkup_date",
            "future_checkup_date",
            "label",
            "selected_transition",
            "full_transition",
            "interval_days",
            "checkup_date",
            "birthday",
            "gender",
            "성별",
        }
    ) | frozenset(QualitativeLabEncoder.QUALITATIVE_COLUMNS)

    def __init__(self, min_numeric_ratio: float = 0.8) -> None:
        """
        Args:
            min_numeric_ratio: 비결측 값 중 수치로 해석 가능한 비율 하한 (0.8 = 80%)
        """
        if not 0 < min_numeric_ratio <= 1:
            raise ValueError("min_numeric_ratio는 0 초과 1 이하여야 합니다.")
        self.min_numeric_ratio = min_numeric_ratio
        self._coercion_log: list[NumericCoercionRecord] = []

    @property
    def coercion_log(self) -> list[NumericCoercionRecord]:
        """마지막 transform()의 변환 요약."""
        return list(self._coercion_log)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """대부분 수치인 컬럼의 텍스트를 공백 처리하고 numeric dtype으로 변환한다."""
        self._coercion_log = []
        result = df.copy()
        exclude = self.DEFAULT_EXCLUDE_COLUMNS

        for col in result.columns:
            if col in exclude:
                continue

            series = result[col]
            non_null = series.notna()
            if not non_null.any():
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            numeric_count = int((non_null & numeric.notna()).sum())
            non_null_count = int(non_null.sum())
            ratio = numeric_count / non_null_count

            text_mask = non_null & numeric.isna()
            text_count = int(text_mask.sum())
            if ratio < self.min_numeric_ratio or text_count == 0:
                continue

            excluded = self._format_value_counts(series.loc[text_mask].astype(str))
            dtype_before = str(series.dtype)
            result[col] = numeric
            dtype_after = str(result[col].dtype)
            self._coercion_log.append(
                NumericCoercionRecord(
                    field=col,
                    numeric_ratio=round(ratio, 4),
                    blanked_count=text_count,
                    excluded_values=excluded,
                    dtype_before=dtype_before,
                    dtype_after=dtype_after,
                )
            )

        return result

    def coercion_field_reports(self) -> list[str]:
        """transform()에서 numeric으로 변환된 인자명 목록."""
        return [r.field for r in self._coercion_log]

    def field_coercion_lines(self, field: str) -> dict[str, str]:
        """인자별 텍스트(공백) 내역·dtype 변환을 한 줄 문자열 dict로 반환한다."""
        record = next((r for r in self._coercion_log if r.field == field), None)
        if record is None:
            raise KeyError(f"변환 이력에 없는 인자입니다: {field}")
        return {
            "텍스트(공백)": record.excluded_values,
            "dtype": f"{record.dtype_before} → {record.dtype_after}",
            "수치비율": (
                f"{record.numeric_ratio * 100:.1f}% "
                f"(텍스트 {record.blanked_count}건 공백)"
            ),
        }

    def coercion_summary(self) -> pd.DataFrame:
        """transform() 직후 호출 가능한 요약 DataFrame."""
        if not self._coercion_log:
            return pd.DataFrame(
                columns=[
                    "field",
                    "numeric_ratio",
                    "blanked_count",
                    "dtype_before",
                    "dtype_after",
                    "excluded_values",
                ]
            )
        return pd.DataFrame(
            {
                "field": r.field,
                "numeric_ratio": r.numeric_ratio,
                "blanked_count": r.blanked_count,
                "dtype_before": r.dtype_before,
                "dtype_after": r.dtype_after,
                "excluded_values": r.excluded_values,
            }
            for r in self._coercion_log
        )

    @staticmethod
    def _format_value_counts(values: pd.Series) -> str:
        """값별 건수를 `내용 (건수)` 형식으로 모두 나열한다."""
        if values.empty:
            return ""

        counts = values.value_counts().sort_values(ascending=False)
        return ", ".join(f"{label} ({int(count)})" for label, count in counts.items())
