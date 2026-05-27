from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LabEncodingRecord:
    """단일 검사 인자 인코딩 요약."""

    field: str
    encoding: str
    negative_count: int
    positive_count: int
    blank_count: int
    missing_count: int


class QualitativeLabEncoder:
    """요검사 정성 결과를 음성/양성으로, B형간염 결과를 수치로 변환한다."""

    URINE_COLUMNS: tuple[str, ...] = (
        "Bilirubin",
        "Blood",
        "Glucose",
        "Keton",
        "Leukocyte",
        "Nitrite",
        "Protein",
        "Urobilinogen",
    )
    HBS_COLUMNS: tuple[str, ...] = ("HBs-Ab", "HBs-Ag")
    QUALITATIVE_COLUMNS: tuple[str, ...] = URINE_COLUMNS + HBS_COLUMNS
    NEGATIVE_LABEL = "음성"
    POSITIVE_LABEL = "양성"
    AUDIT_META_COLUMNS: tuple[str, ...] = (
        "user_key",
        "label",
        "selected_transition",
        "full_transition",
    )

    _NEGATIVE_EXACT: frozenset[str] = frozenset({
        "음성",
        "움성",
        "negative",
        "neg",
        "-",
        "normal",
        "정상",
        "음성(-)",
        "음성(negative)",
    })

    def __init__(self) -> None:
        self._encoding_log: list[LabEncodingRecord] = []
        self._audit_df: pd.DataFrame | None = None
        self._field_mappings: dict[str, pd.DataFrame] = {}

    @property
    def encoding_log(self) -> list[LabEncodingRecord]:
        """마지막 transform()의 인코딩 요약."""
        return list(self._encoding_log)

    def transform(
        self,
        df: pd.DataFrame,
        pre_values: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """요검사·B형간염 컬럼을 변환한다. 결측 컬럼 제거는 하지 않는다.

        Args:
            df: 변환 대상 DataFrame
            pre_values: audit용 원본 값(`{컬럼}_pre`). None이면 transform 직전 df에서 스냅샷.
        """
        self._encoding_log = []
        self._field_mappings = {}
        pre_snapshot = (
            {col: series.copy(deep=True) for col, series in pre_values.items()}
            if pre_values is not None
            else self._capture_pre_values(df)
        )
        result = df.copy()

        for col in self.QUALITATIVE_COLUMNS:
            if col not in result.columns:
                continue
            result[col] = result[col].map(self._encode_qualitative)
            self._append_log(col, "음성/양성", pre_snapshot[col], result[col])
            self._field_mappings[col] = self._build_field_mapping(
                pre_snapshot[col], result[col]
            )

        self._audit_df = self._build_audit_dataframe(df, pre_snapshot, result)
        return result

    def capture_pre_values(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """변환 전 원본 값 스냅샷을 반환한다 (audit `{컬럼}_pre`용)."""
        return self._capture_pre_values(df)

    def build_audit_dataframe(self) -> pd.DataFrame:
        """변환 전·후를 나란히 담은 확인용 DataFrame을 반환한다."""
        if self._audit_df is None:
            raise RuntimeError("transform()를 먼저 호출해야 합니다.")
        return self._audit_df.copy()

    def export_audit(self, output_dir: Path, filename: str) -> Path:
        """변환 확인용 xlsx를 저장한다."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        self.build_audit_dataframe().to_excel(path, index=False)
        return path

    def _build_audit_dataframe(
        self,
        df: pd.DataFrame,
        pre_snapshot: dict[str, pd.Series],
        encoded: pd.DataFrame,
    ) -> pd.DataFrame:
        """메타 컬럼과 `{field}_pre`, `{field}` 쌍으로 확인용 표를 만든다."""
        audit = pd.DataFrame(index=df.index)
        for meta_col in self.AUDIT_META_COLUMNS:
            if meta_col in df.columns:
                audit[meta_col] = df[meta_col]

        for col in (*self.URINE_COLUMNS, *self.HBS_COLUMNS):
            if col not in pre_snapshot:
                continue
            audit[f"{col}_pre"] = pre_snapshot[col]
            audit[col] = encoded[col]

        return audit

    def field_mapping_reports(self) -> dict[str, pd.DataFrame]:
        """인자별 원본값→변환값 매핑과 건수를 반환한다."""
        if not self._field_mappings:
            raise RuntimeError("transform()를 먼저 호출해야 합니다.")
        return {field: report.copy() for field, report in self._field_mappings.items()}

    def field_mapping_lines(self, field: str) -> dict[str, str]:
        """인자별 변환 결과를 양성/음성/NaN 그룹의 한 줄 문자열로 반환한다."""
        if field not in self._field_mappings:
            raise KeyError(f"변환 이력에 없는 인자입니다: {field}")
        return {
            label: self._format_outcome_line(self._field_mappings[field], label)
            for label in (self.POSITIVE_LABEL, self.NEGATIVE_LABEL, "NaN")
        }

    @classmethod
    def _format_outcome_line(cls, mapping: pd.DataFrame, outcome: str) -> str:
        """변환 결과 그룹별 원본값 목록을 한 줄 문자열로 만든다."""
        if outcome == "NaN":
            subset = mapping[mapping["value_after"].isna()]
        else:
            subset = mapping[mapping["value_after"] == outcome]

        if subset.empty:
            return "(없음)"

        parts: list[str] = []
        for _, row in subset.sort_values("count", ascending=False).iterrows():
            before = row["value_before"]
            label = "NaN" if pd.isna(before) else str(before)
            parts.append(f"{label} ({int(row['count'])})")
        return ", ".join(parts)

    @staticmethod
    def _build_field_mapping(before: pd.Series, after: pd.Series) -> pd.DataFrame:
        """원본·변환 쌍별 건수 표를 만든다."""
        pairs = pd.DataFrame(
            {
                "value_before": before,
                "value_after": after,
            }
        )
        report = (
            pairs.groupby(["value_before", "value_after"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        return report

    def encoding_summary(self) -> pd.DataFrame:
        """인코딩 결과 요약표를 반환한다."""
        if not self._encoding_log:
            return pd.DataFrame(
                columns=[
                    "field",
                    "encoding",
                    "negative_count",
                    "positive_count",
                    "blank_count",
                    "missing_count",
                ]
            )
        return pd.DataFrame(
            {
                "field": r.field,
                "encoding": r.encoding,
                "negative_count": r.negative_count,
                "positive_count": r.positive_count,
                "blank_count": r.blank_count,
                "missing_count": r.missing_count,
            }
            for r in self._encoding_log
        )

    def _capture_pre_values(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """변환 직전 원본 값을 깊은 복사로 보관한다."""
        snapshot: dict[str, pd.Series] = {}
        for col in (*self.URINE_COLUMNS, *self.HBS_COLUMNS):
            if col not in df.columns:
                continue
            snapshot[col] = pd.Series(
                df[col].to_numpy(copy=True),
                index=df.index,
                name=col,
            )
        return snapshot

    def _append_log(
        self,
        field: str,
        encoding: str,
        before: pd.Series,
        after: pd.Series,
    ) -> None:
        """인코딩 통계를 로그에 추가한다."""
        negative_count = int((after == self.NEGATIVE_LABEL).sum())
        positive_count = int((after == self.POSITIVE_LABEL).sum())
        blank_count = int(after.isna().sum())
        missing_count = int(before.isna().sum())

        self._encoding_log.append(
            LabEncodingRecord(
                field=field,
                encoding=encoding,
                negative_count=negative_count,
                positive_count=positive_count,
                blank_count=blank_count,
                missing_count=missing_count,
            )
        )

    @classmethod
    def _encode_qualitative(cls, value: object) -> str | float:
        """정성 검사·B형간염 값을 음성 또는 양성으로 변환한다. 분류 불가는 결측."""
        if pd.isna(value):
            return pd.NA
        raw = str(value).strip()
        if not raw:
            return pd.NA

        norm = raw.lower().replace(" ", "")
        compact = re.sub(r"\s+", "", raw.lower())

        if compact in cls._NEGATIVE_EXACT or norm in cls._NEGATIVE_EXACT:
            return cls.NEGATIVE_LABEL

        if cls._is_qualitative_negative(compact, norm, raw):
            return cls.NEGATIVE_LABEL

        if cls._is_qualitative_positive(compact, norm, raw):
            return cls.POSITIVE_LABEL

        return pd.NA

    @classmethod
    def _is_qualitative_negative(cls, compact: str, norm: str, raw: str) -> bool:
        """음성 패턴 여부 (요검사 + B형간염)."""
        if "항체없음" in compact:
            return True
        if "negative" in compact:
            return True
        if "(음성)" in compact or re.search(r"\(음성\)", raw):
            return True
        if compact.startswith("음(") or re.search(r"^음\s*\(", raw):
            return True
        if re.search(r"\(neg\)", raw, re.IGNORECASE):
            return True
        if re.search(r"\(negative\)", raw, re.IGNORECASE):
            return True
        if re.match(r"^neg[\(\s.\d]", raw, re.IGNORECASE):
            return True
        if re.match(r"^negative\s*[\(\s.\d]", raw, re.IGNORECASE):
            return True
        if re.fullmatch(r"negative", norm, re.IGNORECASE):
            return True
        if compact.endswith("음성"):
            return True
        if compact.startswith("-(") or re.search(r"^-\s*\([\d.]", raw):
            return True
        if compact in {"-", "음성", "움성", "negative", "neg", "normal", "정상"}:
            return True
        if compact.startswith("움성"):
            return True
        if norm == "neg" or re.fullmatch(r"neg", norm, re.IGNORECASE):
            return True
        if compact.startswith("음성") and "양성" not in compact:
            return True
        if norm.startswith("negative") and "positive" not in norm:
            return True
        if compact in {"음성(-)", "음성(negative)"}:
            return True
        if re.search(r"^<\s*[\d.]", raw) or compact.startswith("<"):
            return True
        return False

    @classmethod
    def _is_qualitative_positive(cls, compact: str, norm: str, raw: str) -> bool:
        """양성 패턴 여부 (요검사 + B형간염)."""
        if "항체있음" in compact:
            return True
        if "(양성)" in compact or re.search(r"\(양성\)", raw):
            return True
        if re.search(r"\(pos\)", raw, re.IGNORECASE):
            return True
        if re.search(r"\(positive\)", raw, re.IGNORECASE):
            return True
        if re.match(r"^pos\s*[\(\s.\d]", raw, re.IGNORECASE):
            return True
        if re.match(r"^positive\s*[\(\s.\d]", raw, re.IGNORECASE):
            return True
        if re.fullmatch(r"positive", norm, re.IGNORECASE):
            return True
        if compact.startswith("+(") or re.search(r"\+\s*\([\d.]+\)", raw):
            return True
        if compact.endswith("양성"):
            return True
        positive_tokens = (
            "양성",
            "positive",
            "trace",
            "약양성",
            "weaklypositive",
        )
        if any(token in compact for token in positive_tokens):
            return True
        if "weakly" in compact and "positive" in compact:
            return True
        if (
            re.search(r"\d\s*\+", raw)
            or re.search(r"\+\s*\d", raw)
            or re.fullmatch(r"\+\d+", compact)
            or "++" in compact
            or compact.endswith("+")
        ):
            return True
        if compact in {"+", "+-", "+/-", "+1", "+2", "+3", "+4"}:
            return True
        if "1positive" in compact.replace(" ", ""):
            return True
        return False
