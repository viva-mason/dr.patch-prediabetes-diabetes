from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BlankingRecord:
    """단일 인자의 공백 처리 요약."""

    field: str
    below_min: int
    above_max: int
    sentinel: int
    logical: int
    total_blanked: int
    excluded_values: str


class ClinicalRangeBlanker:
    """생리학적으로 나올 수 있는 범위를 최대한 넓게 두고, 불가능·명백한 오류만 공백 처리한다.

    최대 보수 기준: 정상 참고범위·일반적 임상 상한을 적용하지 않는다.
    경계성·고위험 비정상 수치는 유지하고, 센티널·단위 오류·생리학적 불가능 극단만 제거한다.
    """

    SENTINEL_VALUES: frozenset[float] = frozenset({999.0, 999.9, 9999.0, 99999.0})

    DEFAULT_RULES: dict[str, tuple[float, float]] = {
        # 인구·체형
        "나이": (0, 150),
        "신장": (50, 250),
        "체중": (10, 500),
        "BMI": (5, 120),
        "비만도": (0, 500),
        "허리둘레": (10, 500),
        # 혈압 (mmHg)
        "혈압(수축기)": (50, 250),
        "혈압(이완기)": (30, 150),
        # 당뇨·지질 (mg/dL)
        "공복혈당": (1, 1000),
        "T.Cholesterol": (30, 1000),
        "HDL": (1, 300),
        "LDL": (1, 600),
        "Triglyceride": (1, 10000),
        # 간
        "GOT(AST)": (0, 10000),
        "GPT(ALT)": (0, 10000),
        "r-GTP": (0, 5000),
        "ALP": (0, 2000),
        "T.Bilirubin": (0, 50),
        "D.Bilirubin": (0, 30),
        # 신장
        "Creatinine": (0.1, 30),
        "e-GFR": (0, 500),
        "BUN": (0.5, 200),
        "Uric acid": (0.1, 30),
        # 일반혈액 (WBC: K/uL, RBC: M/uL)
        "WBC": (0.1, 200),
        "RBC": (0.5, 15),
        "Hgb": (1, 30),
        "Hct": (5, 90),
        "Platelet": (1, 5000),
        "MCV": (30, 150),
        "MCH": (10, 100),
        "MCHC": (20, 50),
        "RDW": (3, 60),
        "MPV": (3, 25),
        "PDW": (5, 80),
        "Lymphocyte": (0, 100),
        "Monocyte": (0, 100),
        "Eosinophil": (0, 100),
        "Basophil": (0, 100),
        "B/C ratio": (0, 100),
        # 소변
        "PH": (3, 14),
        "SG": (1.000, 1.060),
        # 기타
        "TSH": (0, 500),
        "T.Protein": (2, 15),
        "Albumin": (1, 8),
        "Globulin": (0.5, 10),
        "A/G ratio": (0.1, 25),
    }

    def __init__(self, rules: dict[str, tuple[float, float]] | None = None) -> None:
        """
        Args:
            rules: 인자명 → (최소, 최대) 허용 범위. None이면 DEFAULT_RULES 사용.
        """
        self.rules = rules if rules is not None else dict(self.DEFAULT_RULES)
        self._blanking_log: list[BlankingRecord] = []

    @property
    def blanking_log(self) -> list[BlankingRecord]:
        """마지막 transform()에서 공백 처리된 내역."""
        return list(self._blanking_log)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """범위 밖·센티널·논리 오류 수치를 결측으로 바꾼 DataFrame을 반환한다."""
        result = df.copy()
        self._blanking_log = []

        for field, (min_val, max_val) in self.rules.items():
            if field not in result.columns:
                continue

            series = pd.to_numeric(result[field], errors="coerce")
            if series.notna().sum() == 0:
                continue

            valid = series.notna()
            below = valid & (series < min_val)
            above = valid & (series > max_val)
            sentinel = valid & series.isin(self.SENTINEL_VALUES)
            invalid = below | above | sentinel

            below_count = int(below.sum())
            above_count = int(above.sum())
            sentinel_count = int(sentinel.sum())
            total_blanked = int(invalid.sum())
            excluded_values = ""
            if total_blanked > 0:
                excluded_values = self._format_excluded_values(
                    series.loc[invalid].copy()
                )
                result.loc[invalid, field] = pd.NA
                self._blanking_log.append(
                    BlankingRecord(
                        field=field,
                        below_min=below_count,
                        above_max=above_count,
                        sentinel=sentinel_count,
                        logical=0,
                        total_blanked=total_blanked,
                        excluded_values=excluded_values,
                    )
                )

        logical_count, logical_excluded = self._blank_blood_pressure(result)
        if logical_count:
            self._blanking_log.append(
                BlankingRecord(
                    field="혈압(수축기·이완기)",
                    below_min=0,
                    above_max=0,
                    sentinel=0,
                    logical=logical_count,
                    total_blanked=logical_count,
                    excluded_values=logical_excluded,
                )
            )

        return result

    def blanking_summary(self) -> pd.DataFrame:
        """공백 처리 요약표를 반환한다."""
        columns = [
            "field",
            "below_min",
            "above_max",
            "sentinel",
            "logical",
            "total_blanked",
            "excluded_values",
        ]
        if not self._blanking_log:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            {
                "field": r.field,
                "below_min": r.below_min,
                "above_max": r.above_max,
                "sentinel": r.sentinel,
                "logical": r.logical,
                "total_blanked": r.total_blanked,
                "excluded_values": r.excluded_values,
            }
            for r in self._blanking_log
        ).sort_values("total_blanked", ascending=False).reset_index(drop=True)

    @staticmethod
    def _format_excluded_values(values: pd.Series) -> str:
        """제외된 원본값을 `값 (건수)` 형식으로 모두 나열한다."""
        if values.empty:
            return ""

        counts = values.value_counts().sort_values(ascending=False)
        parts: list[str] = []
        for value, count in counts.items():
            parts.append(f"{ClinicalRangeBlanker._format_scalar(value)} ({int(count)})")
        return ", ".join(parts)

    @staticmethod
    def _format_scalar(value: object) -> str:
        """요약 표시용 스칼라 문자열."""
        if pd.isna(value):
            return "NaN"
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)

    def _blank_blood_pressure(self, df: pd.DataFrame) -> tuple[int, str]:
        """이완기 ≥ 수축기인 경우 두 혈압 값을 결측으로 처리한다."""
        sys_col, dia_col = "혈압(수축기)", "혈압(이완기)"
        if sys_col not in df.columns or dia_col not in df.columns:
            return 0, ""

        systolic = pd.to_numeric(df[sys_col], errors="coerce")
        diastolic = pd.to_numeric(df[dia_col], errors="coerce")
        invalid = systolic.notna() & diastolic.notna() & (diastolic >= systolic)
        count = int(invalid.sum())
        if not count:
            return 0, ""

        pairs = pd.Series(
            (
                f"{self._format_scalar(systolic.loc[idx])}/"
                f"{self._format_scalar(diastolic.loc[idx])}"
                for idx in df.index[invalid]
            ),
            dtype="object",
        )
        excluded_values = self._format_excluded_values(pairs)
        df.loc[invalid, [sys_col, dia_col]] = pd.NA
        return count, excluded_values
