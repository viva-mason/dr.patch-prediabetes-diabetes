from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from core.qualitative_lab_encoder import QualitativeLabEncoder


@dataclass
class NumericCoercionRecord:
    """대부분 수치인 컬럼의 텍스트 공백·numeric 변환 요약."""

    field: str
    numeric_ratio: float
    extracted_count: int
    blanked_count: int
    extracted_values: str
    excluded_values: str
    dtype_before: str
    dtype_after: str


class MostlyNumericCoercer:
    """비결측 값의 대부분이 수치로 해석되는 컬럼에서 텍스트만 공백 처리하고 numeric으로 변환한다.

    "174.1 Cm", "59.8 Kg"처럼 숫자 뒤에 단위 문자열이 붙은 값은 수치 부분만 추출하여 보존한다.
    추출이 불가능한 텍스트만 공백(NaN) 처리한다.
    """

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

    # "174.1 Cm", "59.8Kg", "23.21(과체중)" 등 숫자 + 단위/괄호 접미사 패턴
    _UNIT_SUFFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[A-Za-z(]")

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
        """대부분 수치인 컬럼의 텍스트를 처리하고 numeric dtype으로 변환한다.

        * 숫자+단위 형태("174.1 Cm", "59.8 Kg")는 수치 부분을 추출해 보존한다.
        * 그 외 순수 텍스트는 공백(NaN) 처리한다.
        """
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

            # 숫자+단위 패턴에서 수치 부분 추출 (e.g. "174.1 Cm" → 174.1)
            text_series = series.loc[text_mask].astype(str)
            extracted_str = text_series.str.extract(self._UNIT_SUFFIX_RE)[0]
            extracted_numeric = pd.to_numeric(extracted_str, errors="coerce")

            extractable_idx = extracted_numeric.dropna().index
            extracted_count = len(extractable_idx)
            blanked_count = text_count - extracted_count

            numeric_updated = numeric.copy()
            if extracted_count > 0:
                numeric_updated.loc[extractable_idx] = extracted_numeric.loc[extractable_idx]

            blanked_idx = text_series.index.difference(extractable_idx)
            extracted_vals = self._format_extraction_map(
                series.loc[extractable_idx], extracted_numeric.loc[extractable_idx]
            )
            excluded = self._format_value_counts(series.loc[blanked_idx].astype(str))

            dtype_before = str(series.dtype)
            result[col] = numeric_updated
            dtype_after = str(result[col].dtype)
            self._coercion_log.append(
                NumericCoercionRecord(
                    field=col,
                    numeric_ratio=round(ratio, 4),
                    extracted_count=extracted_count,
                    blanked_count=blanked_count,
                    extracted_values=extracted_vals,
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
        """인자별 추출·공백 내역·dtype 변환을 한 줄 문자열 dict로 반환한다."""
        record = next((r for r in self._coercion_log if r.field == field), None)
        if record is None:
            raise KeyError(f"변환 이력에 없는 인자입니다: {field}")
        return {
            "단위추출(보존)": (
                f"{record.extracted_count}건: {record.extracted_values}"
                if record.extracted_count
                else "없음"
            ),
            "텍스트(공백)": record.excluded_values if record.blanked_count else "없음",
            "dtype": f"{record.dtype_before} → {record.dtype_after}",
            "수치비율": (
                f"{record.numeric_ratio * 100:.1f}% "
                f"(추출 {record.extracted_count}건, 공백 {record.blanked_count}건)"
            ),
        }

    def coercion_summary(self) -> pd.DataFrame:
        """transform() 직후 호출 가능한 요약 DataFrame."""
        if not self._coercion_log:
            return pd.DataFrame(
                columns=[
                    "field",
                    "numeric_ratio",
                    "extracted_count",
                    "blanked_count",
                    "dtype_before",
                    "dtype_after",
                    "extracted_values",
                    "excluded_values",
                ]
            )
        return pd.DataFrame(
            {
                "field": r.field,
                "numeric_ratio": r.numeric_ratio,
                "extracted_count": r.extracted_count,
                "blanked_count": r.blanked_count,
                "dtype_before": r.dtype_before,
                "dtype_after": r.dtype_after,
                "extracted_values": r.extracted_values,
                "excluded_values": r.excluded_values,
            }
            for r in self._coercion_log
        )

    @staticmethod
    def _format_extraction_map(original: pd.Series, extracted: pd.Series) -> str:
        """원본 텍스트 → 추출 수치 매핑을 `원본 → 수치 (N건)` 형식으로 나열한다."""
        if original.empty:
            return ""
        df = pd.DataFrame({"orig": original.astype(str).values, "val": extracted.values})
        summary = (
            df.groupby("orig", sort=False)["val"]
            .agg(val="first", count="count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        return ", ".join(
            f"{row.orig} → {row.val} ({int(row.count)}건)"
            for row in summary.itertuples()
        )

    @staticmethod
    def _format_value_counts(values: pd.Series) -> str:
        """값별 건수를 `내용 (건수)` 형식으로 모두 나열한다."""
        if values.empty:
            return ""

        counts = values.value_counts().sort_values(ascending=False)
        return ", ".join(f"{label} ({int(count)})" for label, count in counts.items())
