from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from core.clinical_range_blanker import ClinicalRangeBlanker


@dataclass
class ImputationRecord:
    """단일 임상 대체 규칙의 적용 결과."""

    field: str
    filled_count: int
    rule: str


class ClinicalPreprocessor:
    """건강검진 전이 데이터셋에 임상적으로 유도 가능한 결측을 대체한다.

    입력·계산에 쓰이는 수치에 센티널(999 등)이 있으면 해당 행의 대체는 수행하지 않는다.
    """

    SENTINEL_VALUES: frozenset[float] = ClinicalRangeBlanker.SENTINEL_VALUES

    SYS_COL = "혈압(수축기)"
    DIA_COL = "혈압(이완기)"
    WBC_COL = "WBC"
    WBC_PER_UL_THRESHOLD = 1000.0
    RBC_COL = "RBC"
    RBC_PER_UL_THRESHOLD = 100.0
    MCH_COL = "MCH"
    MCV_COL = "MCV"
    CBC_SWAP_TOLERANCE = 0.5
    CBC_SWAP_REQUIRED_COLUMNS: tuple[str, ...] = (
        "RBC",
        "Hgb",
        "Hct",
        "MCV",
        "MCH",
        "MCHC",
    )
    GENDER_COL = "gender"
    SEX_TEXT_COL = "성별"
    MALE_LABEL = "남자"
    FEMALE_LABEL = "여자"

    _MALE_TEXT = frozenset({"M", "남", "남자", "MALE", "MAN"})
    _FEMALE_TEXT = frozenset({"F", "여", "여자", "FEMALE", "WOMAN", "W"})

    IMPUTATION_TARGET_COLUMNS: tuple[str, ...] = (
        "나이",
        "BMI",
        "T.Cholesterol",
        "HDL",
        "Triglyceride",
        "LDL",
        "Globulin",
        "Albumin",
        "T.Protein",
        "A/G ratio",
        "UIBC",
        "철포화율",
        "비만도",
        "e-GFR",
        "MCH",
        "MCHC",
        "MCV",
        "Hct",
    )

    def __init__(self) -> None:
        self._imputation_log: list[ImputationRecord] = []
        self._bp_swap_pairs: pd.Series = pd.Series(dtype="object")
        self._wbc_fix_pairs: pd.Series = pd.Series(dtype="object")
        self._rbc_fix_pairs: pd.Series = pd.Series(dtype="object")
        self._mch_mcv_swap_pairs: pd.Series = pd.Series(dtype="object")

    @property
    def imputation_log(self) -> list[ImputationRecord]:
        """마지막 transform()에서 적용된 대체 내역."""
        return list(self._imputation_log)

    def fix_blood_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """혈압 반대 기입(이완기≥수축기)만 교환한다."""
        self._bp_swap_pairs = pd.Series(dtype="object")
        return self._fix_swapped_blood_pressure(df.copy())

    def fix_wbc_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """WBC가 /µL로 입력된 값(≥1000)을 K/µL로 보정한다 (÷1000, 소수 둘째 자리)."""
        self._wbc_fix_pairs = pd.Series(dtype="object")
        if self.WBC_COL not in df.columns:
            return df.copy()

        result = df.copy()
        wbc = self._without_sentinels(result[self.WBC_COL])
        needs_fix = wbc.notna() & (wbc >= self.WBC_PER_UL_THRESHOLD)
        if not needs_fix.any():
            return result

        corrected = (wbc.loc[needs_fix] / 1000).round(2)
        self._wbc_fix_pairs = pd.Series(
            (
                f"{self._format_scalar(wbc.loc[idx])} → {corrected.loc[idx]:.2f}"
                for idx in result.index[needs_fix]
            ),
            dtype="object",
        )
        result.loc[needs_fix, self.WBC_COL] = corrected.to_numpy()
        return result

    def fix_rbc_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """RBC가 /µL로 입력된 값(≥100)을 M/µL로 보정한다 (÷100, 소수 둘째 자리)."""
        self._rbc_fix_pairs = pd.Series(dtype="object")
        if self.RBC_COL not in df.columns:
            return df.copy()

        result = df.copy()
        rbc = self._without_sentinels(result[self.RBC_COL])
        needs_fix = rbc.notna() & (rbc >= self.RBC_PER_UL_THRESHOLD)
        if not needs_fix.any():
            return result

        corrected = (rbc.loc[needs_fix] / 100).round(2)
        self._rbc_fix_pairs = pd.Series(
            (
                f"{self._format_scalar(rbc.loc[idx])} → {corrected.loc[idx]:.2f}"
                for idx in result.index[needs_fix]
            ),
            dtype="object",
        )
        result.loc[needs_fix, self.RBC_COL] = corrected.to_numpy()
        return result

    def fix_mch_mcv_swap(self, df: pd.DataFrame) -> pd.DataFrame:
        """MCH·MCV 컬럼 뒤바뀜(6개 CBC 지표 모두 유효)을 감지해 두 값을 교환한다."""
        self._mch_mcv_swap_pairs = pd.Series(dtype="object")
        if not all(col in df.columns for col in self.CBC_SWAP_REQUIRED_COLUMNS):
            return df.copy()

        result = df.copy()
        rbc = self._without_sentinels(result[self.RBC_COL])
        hgb = self._without_sentinels(result["Hgb"])
        hct = self._without_sentinels(result["Hct"])
        mcv = self._without_sentinels(result[self.MCV_COL])
        mch = self._without_sentinels(result[self.MCH_COL])

        complete = (
            rbc.notna()
            & hgb.notna()
            & hct.notna()
            & mcv.notna()
            & mch.notna()
            & self._without_sentinels(result["MCHC"]).notna()
            & (rbc != 0)
        )
        mch_expected = hgb / rbc * 10
        mcv_expected = hct / rbc * 10
        tol = self.CBC_SWAP_TOLERANCE

        normal = (
            complete
            & (mch.sub(mch_expected).abs() <= tol)
            & (mcv.sub(mcv_expected).abs() <= tol)
        )
        inverted = (
            complete
            & ~normal
            & (mcv.sub(mch_expected).abs() <= tol)
            & (mch.sub(mcv_expected).abs() <= tol)
        )
        if not inverted.any():
            return result

        mch_before = result.loc[inverted, self.MCH_COL]
        mcv_before = result.loc[inverted, self.MCV_COL]
        result.loc[inverted, self.MCH_COL] = mcv_before.to_numpy()
        result.loc[inverted, self.MCV_COL] = mch_before.to_numpy()

        self._mch_mcv_swap_pairs = pd.Series(
            (
                f"MCH {self._format_scalar(mch_before.loc[idx])}→{self._format_scalar(mcv_before.loc[idx])}"
                f" / MCV {self._format_scalar(mcv_before.loc[idx])}→{self._format_scalar(mch_before.loc[idx])}"
                for idx in result.index[inverted]
            ),
            dtype="object",
        )
        return result

    def transform_imputations(self, df: pd.DataFrame) -> pd.DataFrame:
        """나이·BMI·지질·e-GFR·일반혈액 등 임상 대체를 적용한다 (혈압 교환 제외)."""
        result = df.copy()
        self._imputation_log = []

        result = self._impute_age(result)
        result = self._unify_gender(result)
        result = self._impute_bmi(result)
        result = self._impute_lipid_panel(result)
        result = self._impute_globulin(result)
        result = self._impute_albumin(result)
        result = self._impute_total_protein(result)
        result = self._impute_ag_ratio(result)
        result = self._impute_uibc(result)
        result = self._impute_iron_saturation(result)
        result = self._impute_obesity_degree(result)
        result = self._impute_egfr(result)
        result = self._impute_cbc_indices(result)

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """혈압 교환 후 임상 대체까지 전처리 파이프라인을 적용한다."""
        return self.transform_imputations(self.fix_blood_pressure(df))

    def imputation_summary(self) -> pd.DataFrame:
        """적용된 대체 규칙 요약을 반환한다."""
        if not self._imputation_log:
            return pd.DataFrame(columns=["field", "filled_count", "rule"])
        return pd.DataFrame(
            {
                "field": [r.field for r in self._imputation_log],
                "filled_count": [r.filled_count for r in self._imputation_log],
                "rule": [r.rule for r in self._imputation_log],
            }
        )

    def blood_pressure_swap_summary(self) -> pd.DataFrame:
        """혈압 반대 기입 교환 내역 요약을 반환한다."""
        columns = ["field", "swapped_count", "swap_detail"]
        if self._bp_swap_pairs.empty:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            {
                "field": [f"{self.SYS_COL}·{self.DIA_COL}"],
                "swapped_count": [int(len(self._bp_swap_pairs))],
                "swap_detail": [self._format_value_counts(self._bp_swap_pairs)],
            }
        )

    def wbc_unit_fix_summary(self) -> pd.DataFrame:
        """WBC /µL→K/µL 단위 보정 내역 요약을 반환한다."""
        columns = ["field", "fixed_count", "fix_detail"]
        if self._wbc_fix_pairs.empty:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            {
                "field": [self.WBC_COL],
                "fixed_count": [int(len(self._wbc_fix_pairs))],
                "fix_detail": [self._format_value_counts(self._wbc_fix_pairs)],
            }
        )

    def rbc_unit_fix_summary(self) -> pd.DataFrame:
        """RBC /µL→M/µL 단위 보정 내역 요약을 반환한다."""
        columns = ["field", "fixed_count", "fix_detail"]
        if self._rbc_fix_pairs.empty:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            {
                "field": [self.RBC_COL],
                "fixed_count": [int(len(self._rbc_fix_pairs))],
                "fix_detail": [self._format_value_counts(self._rbc_fix_pairs)],
            }
        )

    def mch_mcv_swap_summary(self) -> pd.DataFrame:
        """MCH·MCV 뒤바뀜 교환 내역 요약을 반환한다."""
        columns = ["field", "swapped_count", "swap_detail"]
        if self._mch_mcv_swap_pairs.empty:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            {
                "field": [f"{self.MCH_COL}·{self.MCV_COL}"],
                "swapped_count": [int(len(self._mch_mcv_swap_pairs))],
                "swap_detail": [self._format_value_counts(self._mch_mcv_swap_pairs)],
            }
        )

    def missing_rate_comparison(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> pd.DataFrame:
        """전처리 대상 컬럼의 결측 건수·비율을 전/후로 비교한다."""
        if len(before) != len(after):
            raise ValueError("before와 after의 행 수가 같아야 합니다.")

        n = len(before)
        rows: list[dict[str, object]] = []
        for col in self.IMPUTATION_TARGET_COLUMNS:
            if col not in before.columns and col not in after.columns:
                continue

            before_series = (
                before[col]
                if col in before.columns
                else pd.Series([pd.NA] * n, index=before.index)
            )
            after_series = (
                after[col]
                if col in after.columns
                else pd.Series([pd.NA] * n, index=after.index)
            )

            missing_before = int(pd.to_numeric(before_series, errors="coerce").isna().sum())
            missing_after = int(pd.to_numeric(after_series, errors="coerce").isna().sum())
            pct_before = round(missing_before / n * 100, 2)
            pct_after = round(missing_after / n * 100, 2)

            rows.append(
                {
                    "field": col,
                    "rows": n,
                    "missing_before": missing_before,
                    "missing_pct_before": pct_before,
                    "missing_after": missing_after,
                    "missing_pct_after": pct_after,
                    "filled": missing_before - missing_after,
                    "pct_point_change": round(pct_after - pct_before, 2),
                }
            )

        return pd.DataFrame(rows)

    def export(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        filename: str,
    ) -> Path:
        """전처리된 DataFrame을 xlsx로 저장한다."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        df.to_excel(path, index=False)
        return path

    def _fix_swapped_blood_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """이완기 ≥ 수축기이면 수축기·이완기 값을 서로 교환한다 (컬럼 반대 기입)."""
        result = df.copy()
        if self.SYS_COL not in result.columns or self.DIA_COL not in result.columns:
            return result

        systolic = self._without_sentinels(result[self.SYS_COL])
        diastolic = self._without_sentinels(result[self.DIA_COL])
        swapped = systolic.notna() & diastolic.notna() & (diastolic >= systolic)
        count = int(swapped.sum())
        if not count:
            return result

        self._bp_swap_pairs = pd.Series(
            (
                f"{self._format_scalar(systolic.loc[idx])}/{self._format_scalar(diastolic.loc[idx])}"
                f" → {self._format_scalar(diastolic.loc[idx])}/{self._format_scalar(systolic.loc[idx])}"
                for idx in result.index[swapped]
            ),
            dtype="object",
        )
        result.loc[swapped, self.SYS_COL] = diastolic.loc[swapped].to_numpy()
        result.loc[swapped, self.DIA_COL] = systolic.loc[swapped].to_numpy()
        return result

    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """나이 결측을 birthday와 checkup_date로 계산하여 채운다."""
        result = df.copy()
        if "나이" not in result.columns:
            return result

        ages = self._without_sentinels(result["나이"])
        derived = self._age_from_dates(result)
        mask = ages.isna() & derived.notna()
        if not mask.any():
            return result

        result.loc[mask, "나이"] = self._round_to_column_precision(
            ages, derived.loc[mask], fallback=0
        )
        self._log(
            "나이",
            int(mask.sum()),
            "나이 = (`checkup_date` - `birthday`).days / 365.25",
        )
        return result

    def _unify_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """gender(1/2)와 성별(M/F 등)을 '남자'/'여자' 단일 컬럼으로 통합한다."""
        result = df.copy()
        from_numeric = self._gender_from_numeric(result.get(self.GENDER_COL))
        from_text = self._gender_from_text(result.get(self.SEX_TEXT_COL))

        unified = from_numeric.copy()
        text_fill = from_numeric.isna() & from_text.notna()
        unified.loc[text_fill] = from_text.loc[text_fill]

        if self.SEX_TEXT_COL in result.columns:
            result = result.drop(columns=[self.SEX_TEXT_COL])

        result[self.GENDER_COL] = unified
        invalid = result[self.GENDER_COL].notna() & ~result[self.GENDER_COL].isin(
            [self.MALE_LABEL, self.FEMALE_LABEL]
        )
        if invalid.any():
            result.loc[invalid, self.GENDER_COL] = pd.NA

        return result

    def _gender_from_numeric(self, series: pd.Series | None) -> pd.Series:
        """숫자 gender(1=남자, 2=여자)를 문자열 레이블로 변환한다."""
        if series is None:
            return pd.Series(pd.NA, index=range(0))

        numeric = pd.to_numeric(series, errors="coerce")
        mapped = pd.Series(pd.NA, index=series.index, dtype="object")
        mapped.loc[numeric == 1] = self.MALE_LABEL
        mapped.loc[numeric == 2] = self.FEMALE_LABEL
        return mapped

    def _gender_from_text(self, series: pd.Series | None) -> pd.Series:
        """성별 텍스트 컬럼을 '남자'/'여자'로 변환한다."""
        if series is None:
            return pd.Series(pd.NA, index=range(0))

        mapped = pd.Series(pd.NA, index=series.index, dtype="object")
        for idx, value in series.items():
            label = self._parse_gender_text(value)
            if label is not None:
                mapped.loc[idx] = label
        return mapped

    def _parse_gender_text(self, value: object) -> str | None:
        """성별 텍스트 값을 남자/여자로 파싱한다. 숫자 등 비정상 값은 None."""
        if pd.isna(value):
            return None

        text = str(value).strip()
        if not text:
            return None

        if text.isdigit():
            return None

        upper = text.upper()
        if upper in self._MALE_TEXT:
            return self.MALE_LABEL
        if upper in self._FEMALE_TEXT:
            return self.FEMALE_LABEL
        return None

    def _impute_bmi(self, df: pd.DataFrame) -> pd.DataFrame:
        """BMI = 체중 / (신장[cm] / 100)² 로 결측을 대체한다."""
        return self._fill_numeric(
            df,
            target="BMI",
            rule="BMI = 체중 / (신장/100)²",
            compute=lambda row: row["체중"] / (row["신장"] / 100) ** 2,
            requires=["체중", "신장"],
            decimal_fallback=1,
        )

    def _impute_lipid_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Friedewald 관계로 T.Cholesterol, HDL, Triglyceride, LDL 결측을 대체한다 (TG < 400)."""
        result = df.copy()
        lipid_cols = ("T.Cholesterol", "HDL", "Triglyceride", "LDL")
        if not all(col in result.columns for col in lipid_cols):
            return result

        tc = self._without_sentinels(result["T.Cholesterol"])
        hdl = self._without_sentinels(result["HDL"])
        tg = self._without_sentinels(result["Triglyceride"])
        ldl = self._without_sentinels(result["LDL"])

        tg_ok = tg.notna() & (tg < 400)
        vldl_from_tg = tg // 5

        mask_ldl = ldl.isna() & tc.notna() & hdl.notna() & tg_ok
        if mask_ldl.any():
            self._assign_imputed(
                result,
                "LDL",
                mask_ldl,
                tc.loc[mask_ldl] - hdl.loc[mask_ldl] - vldl_from_tg.loc[mask_ldl],
            )
            self._log(
                "LDL",
                int(mask_ldl.sum()),
                "LDL = T.Cholesterol - HDL - int(Triglyceride/5), Triglyceride<400",
            )
            ldl = self._without_sentinels(result["LDL"])

        mask_tc = tc.isna() & ldl.notna() & hdl.notna() & tg_ok
        if mask_tc.any():
            self._assign_imputed(
                result,
                "T.Cholesterol",
                mask_tc,
                ldl.loc[mask_tc] + hdl.loc[mask_tc] + vldl_from_tg.loc[mask_tc],
            )
            self._log(
                "T.Cholesterol",
                int(mask_tc.sum()),
                "T.Cholesterol = LDL + HDL + int(Triglyceride/5), Triglyceride<400",
            )
            tc = self._without_sentinels(result["T.Cholesterol"])

        mask_hdl = hdl.isna() & tc.notna() & ldl.notna() & tg_ok
        if mask_hdl.any():
            self._assign_imputed(
                result,
                "HDL",
                mask_hdl,
                tc.loc[mask_hdl] - ldl.loc[mask_hdl] - vldl_from_tg.loc[mask_hdl],
            )
            self._log(
                "HDL",
                int(mask_hdl.sum()),
                "HDL = T.Cholesterol - LDL - int(Triglyceride/5), Triglyceride<400",
            )
            hdl = self._without_sentinels(result["HDL"])

        vldl_est = tc - ldl - hdl
        calc_tg = 5 * vldl_est
        mask_tg = (
            tg.isna()
            & tc.notna()
            & ldl.notna()
            & hdl.notna()
            & (vldl_est >= 0)
            & (calc_tg < 400)
        )
        if mask_tg.any():
            self._assign_imputed(result, "Triglyceride", mask_tg, calc_tg.loc[mask_tg])
            self._log(
                "Triglyceride",
                int(mask_tg.sum()),
                "Triglyceride = 5×(T.Cholesterol-LDL-HDL), Triglyceride<400",
            )

        return result

    def _impute_globulin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Globulin = T.Protein - Albumin."""
        return self._fill_numeric(
            df,
            target="Globulin",
            rule="Globulin = T.Protein - Albumin",
            compute=lambda row: row["T.Protein"] - row["Albumin"],
            requires=["T.Protein", "Albumin"],
        )

    def _impute_albumin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Albumin = T.Protein - Globulin."""
        return self._fill_numeric(
            df,
            target="Albumin",
            rule="Albumin = T.Protein - Globulin",
            compute=lambda row: row["T.Protein"] - row["Globulin"],
            requires=["T.Protein", "Globulin"],
        )

    def _impute_total_protein(self, df: pd.DataFrame) -> pd.DataFrame:
        """T.Protein = Albumin + Globulin."""
        return self._fill_numeric(
            df,
            target="T.Protein",
            rule="T.Protein = Albumin + Globulin",
            compute=lambda row: row["Albumin"] + row["Globulin"],
            requires=["Albumin", "Globulin"],
        )

    def _impute_ag_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """A/G ratio = Albumin / Globulin."""
        return self._fill_numeric(
            df,
            target="A/G ratio",
            rule="A/G ratio = Albumin / Globulin",
            compute=lambda row: row["Albumin"] / row["Globulin"],
            requires=["Albumin", "Globulin"],
            extra_mask=lambda frame: frame["Globulin"] != 0,
        )

    def _impute_uibc(self, df: pd.DataFrame) -> pd.DataFrame:
        """UIBC = TIBC - Fe."""
        return self._fill_numeric(
            df,
            target="UIBC",
            rule="UIBC = TIBC - Fe",
            compute=lambda row: row["TIBC"] - row["Fe"],
            requires=["TIBC", "Fe"],
        )

    def _impute_iron_saturation(self, df: pd.DataFrame) -> pd.DataFrame:
        """철포화율 = Fe / TIBC × 100."""
        return self._fill_numeric(
            df,
            target="철포화율",
            rule="철포화율 = Fe / TIBC × 100",
            compute=lambda row: row["Fe"] / row["TIBC"] * 100,
            requires=["Fe", "TIBC"],
            extra_mask=lambda frame: frame["TIBC"] != 0,
        )

    def _impute_obesity_degree(self, df: pd.DataFrame) -> pd.DataFrame:
        """비만도 = (체중 / 표준체중) × 100, 표준체중 = (신장-100) × (남0.9/여0.85)."""
        result = df.copy()
        if "비만도" not in result.columns:
            return result

        weight = self._without_sentinels(result.get("체중"))
        height = self._without_sentinels(result.get("신장"))
        obesity = self._without_sentinels(result["비만도"])
        coef = result[self.GENDER_COL].map(
            {self.MALE_LABEL: 0.9, self.FEMALE_LABEL: 0.85}
        )

        standard_weight = (height - 100) * coef
        mask = (
            obesity.isna()
            & weight.notna()
            & height.notna()
            & coef.notna()
            & (standard_weight > 0)
        )
        if not mask.any():
            return result

        calculated = weight / standard_weight * 100
        self._assign_imputed(result, "비만도", mask, calculated)
        self._log(
            "비만도",
            int(mask.sum()),
            "비만도 = (체중/표준체중)×100, 표준체중=(신장-100)×(남0.9/여0.85)",
        )
        return result

    def _impute_cbc_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """일반혈액 MCH·MCHC·MCV·Hct 결측을 유도 공식으로 대체한다 (결측만)."""
        result = self._fill_numeric(
            df,
            target="MCH",
            rule="MCH = Hgb / RBC × 10",
            compute=lambda row: row["Hgb"] / row["RBC"] * 10,
            requires=["Hgb", "RBC"],
            extra_mask=lambda frame: frame["RBC"] != 0,
        )
        result = self._fill_numeric(
            result,
            target="MCHC",
            rule="MCHC = Hgb / Hct × 100",
            compute=lambda row: row["Hgb"] / row["Hct"] * 100,
            requires=["Hgb", "Hct"],
            extra_mask=lambda frame: frame["Hct"] != 0,
        )
        result = self._fill_numeric(
            result,
            target="MCV",
            rule="MCV = Hct / RBC × 10",
            compute=lambda row: row["Hct"] / row["RBC"] * 10,
            requires=["Hct", "RBC"],
            extra_mask=lambda frame: frame["RBC"] != 0,
        )
        return self._fill_numeric(
            result,
            target="Hct",
            rule="Hct = RBC × MCV / 10",
            compute=lambda row: row["RBC"] * row["MCV"] / 10,
            requires=["RBC", "MCV"],
        )

    def _impute_egfr(self, df: pd.DataFrame) -> pd.DataFrame:
        """CKD-EPI(2009)로 e-GFR 결측을 대체한다."""
        result = df.copy()
        if "e-GFR" not in result.columns or "Creatinine" not in result.columns:
            return result

        creatinine = self._without_sentinels(result["Creatinine"])
        egfr = self._without_sentinels(result["e-GFR"])
        ages = self._resolve_age(result)

        mask = egfr.isna() & creatinine.notna() & ages.notna() & result[self.GENDER_COL].notna()
        if not mask.any():
            return result

        calculated = pd.Series(index=result.index, dtype="float64")
        for idx in result.index[mask]:
            row = result.loc[idx]
            age = float(ages.loc[idx])
            cr = float(creatinine.loc[idx])
            female = row[self.GENDER_COL] == self.FEMALE_LABEL
            calculated.loc[idx] = self._ckd_epi_2009(cr, age, female)

        self._assign_imputed(result, "e-GFR", mask, calculated)
        self._log("e-GFR", int(mask.sum()), "CKD-EPI(2009), Creatinine·나이·gender")
        return result

    def _age_from_dates(self, df: pd.DataFrame) -> pd.Series:
        """birthday와 checkup_date로 연령(세)을 계산한다."""
        if "birthday" not in df.columns or "checkup_date" not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="float64")

        birthday = pd.to_datetime(df["birthday"], errors="coerce")
        checkup = pd.to_datetime(df["checkup_date"], errors="coerce")
        return (checkup - birthday).dt.days / 365.25

    def _resolve_age(self, df: pd.DataFrame) -> pd.Series:
        """나이 컬럼을 반환하고, 결측이면 birthday·checkup_date로 보완한다."""
        if "나이" in df.columns:
            ages = self._without_sentinels(df["나이"])
        else:
            ages = pd.Series(pd.NA, index=df.index, dtype="float64")

        return ages.fillna(self._age_from_dates(df))

    @staticmethod
    def _format_value_counts(values: pd.Series) -> str:
        """값별 건수를 `내용 (건수)` 형식으로 모두 나열한다."""
        if values.empty:
            return ""

        counts = values.value_counts().sort_values(ascending=False)
        return ", ".join(f"{label} ({int(count)})" for label, count in counts.items())

    @staticmethod
    def _format_scalar(value: object) -> str:
        """요약 표시용 스칼라 문자열."""
        if pd.isna(value):
            return "NaN"
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)

    @classmethod
    def _without_sentinels(cls, series: pd.Series | None) -> pd.Series:
        """센티널 값을 결측으로 간주한 수치 Series를 반환한다."""
        if series is None:
            return pd.Series(dtype="float64")

        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.mask(numeric.isin(cls.SENTINEL_VALUES))

    @staticmethod
    def _ckd_epi_2009(creatinine_mg_dl: float, age: float, female: bool) -> float:
        """CKD-EPI 2009 creatinine equation (race-free)."""
        if female:
            kappa = 0.7
            alpha = -0.329
            sex_factor = 1.018
        else:
            kappa = 0.9
            alpha = -0.411
            sex_factor = 1.0

        ratio = creatinine_mg_dl / kappa
        if creatinine_mg_dl <= kappa:
            return (
                141
                * min(ratio, 1.0) ** alpha
                * max(ratio, 1.0) ** -0.329
                * (0.993 ** age)
                * sex_factor
            )
        return (
            141
            * min(ratio, 1.0) ** -1.209
            * max(ratio, 1.0) ** -1.209
            * (0.993 ** age)
            * sex_factor
        )

    @staticmethod
    def _infer_decimal_places(series: pd.Series, fallback: int = 1) -> int:
        """기존 값의 소수점 자릿수 최빈값을 반환한다."""
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return fallback

        counts: Counter[int] = Counter()
        for value in numeric:
            text = format(float(value), ".12f").rstrip("0").rstrip(".")
            places = len(text.split(".")[1]) if "." in text else 0
            counts[places] += 1
        return counts.most_common(1)[0][0]

    def _round_to_column_precision(
        self,
        reference: pd.Series,
        values: pd.Series,
        fallback: int = 1,
    ) -> pd.Series:
        """reference 컬럼과 동일한 소수점 자릿수로 values를 반올림한다."""
        places = self._infer_decimal_places(reference, fallback=fallback)
        return pd.to_numeric(values, errors="coerce").round(places)

    def _assign_imputed(
        self,
        result: pd.DataFrame,
        column: str,
        mask: pd.Series,
        values: pd.Series,
        decimal_fallback: int = 1,
    ) -> None:
        """대체값을 기존 컬럼 소수점 규칙에 맞춰 기록한다."""
        reference = pd.to_numeric(result[column], errors="coerce")
        rounded = self._round_to_column_precision(
            reference,
            values,
            fallback=decimal_fallback,
        )
        result.loc[mask, column] = rounded

    def _fill_numeric(
        self,
        df: pd.DataFrame,
        target: str,
        rule: str,
        compute: Callable[[pd.Series], float],
        requires: list[str],
        extra_mask: Callable[[pd.DataFrame], pd.Series] | None = None,
        decimal_fallback: int = 1,
    ) -> pd.DataFrame:
        """필수 컬럼이 유효할 때만 target 결측을 계산값으로 채운다."""
        result = df.copy()
        if target not in result.columns:
            return result

        for col in requires:
            if col not in result.columns:
                return result

        frame = pd.DataFrame(
            {col: self._without_sentinels(result[col]) for col in requires},
            index=result.index,
        )
        target_vals = self._without_sentinels(result[target])
        mask = target_vals.isna() & frame.notna().all(axis=1)
        if extra_mask is not None:
            mask &= extra_mask(frame)

        if not mask.any():
            return result

        calculated = frame.loc[mask].apply(compute, axis=1)
        self._assign_imputed(
            result,
            target,
            mask,
            calculated,
            decimal_fallback=decimal_fallback,
        )
        self._log(target, int(mask.sum()), rule)
        return result

    def _log(self, field: str, filled_count: int, rule: str) -> None:
        if filled_count > 0:
            self._imputation_log.append(
                ImputationRecord(field=field, filled_count=filled_count, rule=rule)
            )
