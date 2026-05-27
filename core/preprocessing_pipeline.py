from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.clinical_range_blanker import ClinicalRangeBlanker
from core.column_filter import MissingRateColumnFilter
from core.mostly_numeric_coercer import MostlyNumericCoercer
from core.preprocessor import ClinicalPreprocessor
from core.qualitative_lab_encoder import QualitativeLabEncoder


@dataclass
class PreprocessingConfig:
    """전처리 노트북 설정."""

    input_dir: Path
    output_dir: Path
    missing_rate_threshold: float = 0.20
    min_numeric_ratio: float = 0.80
    extra_preserve_columns: frozenset[str] = field(default_factory=frozenset)
    extra_always_drop_columns: frozenset[str] = field(default_factory=frozenset)
    dataset_paths: dict[str, Path] | None = None
    export_filenames: dict[str, str] | None = None

    def resolve_dataset_paths(self) -> dict[str, Path]:
        """데이터셋명 → 입력 xlsx 경로."""
        if self.dataset_paths is not None:
            return dict(self.dataset_paths)
        return {
            "pre_diabetes": self.input_dir / "pre_diabetes_dataset.xlsx",
            "diabetes": self.input_dir / "diabetes_dataset.xlsx",
        }

    def resolve_export_filenames(self) -> dict[str, str]:
        """데이터셋명 → 출력 xlsx 파일명."""
        if self.export_filenames is not None:
            return dict(self.export_filenames)
        return {
            "pre_diabetes": "pre_diabetes_dataset.xlsx",
            "diabetes": "diabetes_dataset.xlsx",
        }

    def resolve_preserve_columns(self) -> frozenset[str]:
        """고결측 필터에서 항상 유지할 컬럼."""
        return (
            MissingRateColumnFilter.DEFAULT_PRESERVE_COLUMNS | self.extra_preserve_columns
        )


class PreprocessingPipeline:
    """260526_preprocessing 노트북 단계별 전처리."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = ClinicalPreprocessor()
        self.numeric_coercer = MostlyNumericCoercer(
            min_numeric_ratio=config.min_numeric_ratio
        )
        self.lab_encoder = QualitativeLabEncoder()
        self.range_blanker = ClinicalRangeBlanker()
        self.column_filter = self._build_column_filter()
        self.processed: dict[str, pd.DataFrame] = {}
        self.bp_swap_reports: dict[str, pd.DataFrame] = {}
        self.wbc_fix_reports: dict[str, pd.DataFrame] = {}
        self.rbc_fix_reports: dict[str, pd.DataFrame] = {}
        self.mch_mcv_swap_reports: dict[str, pd.DataFrame] = {}
        self.numeric_coercion_reports: dict[str, pd.DataFrame] = {}
        self.lab_encoding_reports: dict[str, pd.DataFrame] = {}
        self.blanking_reports: dict[str, pd.DataFrame] = {}
        self.imputation_reports: dict[str, pd.DataFrame] = {}
        self.missing_reports: dict[str, pd.DataFrame] = {}
        self.dropped_reports: dict[str, pd.DataFrame] = {}

    def configure_column_filter(
        self,
        *,
        extra_preserve_columns: frozenset[str] | None = None,
        extra_always_drop_columns: frozenset[str] | None = None,
        missing_rate_threshold: float | None = None,
    ) -> None:
        """§2 고결측 제외 설정을 갱신한다."""
        if extra_preserve_columns is not None:
            self.config.extra_preserve_columns = extra_preserve_columns
        if extra_always_drop_columns is not None:
            self.config.extra_always_drop_columns = extra_always_drop_columns
        if missing_rate_threshold is not None:
            self.config.missing_rate_threshold = missing_rate_threshold
        self.column_filter = self._build_column_filter()

    def run_early_corrections(self) -> None:
        """수치형 텍스트 공백·numeric 변환, 혈압·WBC/RBC·MCH/MCV 교환 보정."""
        for name, path in self.config.resolve_dataset_paths().items():
            raw_df = pd.read_excel(path)
            df = self.numeric_coercer.transform(raw_df)
            self.numeric_coercion_reports[name] = self.numeric_coercer.coercion_summary()

            print(f"\n[{name}] rows={len(df):,}, cols={df.shape[1]}")
            print(
                f"— 수치형 텍스트 공백 처리 (비결측 중 "
                f"≥{self.config.min_numeric_ratio * 100:.0f}% 수치 해석 가능)"
            )
            if self.numeric_coercion_reports[name].empty:
                print("  변환된 인자 없음")
            else:
                self._display_wide(
                    self.numeric_coercion_reports[name][
                        [
                            "field",
                            "numeric_ratio",
                            "blanked_count",
                            "dtype_before",
                            "dtype_after",
                            "excluded_values",
                        ]
                    ]
                )

            df = self.preprocessor.fix_blood_pressure(df)
            self.bp_swap_reports[name] = self.preprocessor.blood_pressure_swap_summary()

            print("— 혈압 반대 기입 교환 (수축기/이완기)")
            self._show_report(self.bp_swap_reports[name], empty_message="  교환 없음")

            df = self.preprocessor.fix_wbc_unit(df)
            self.wbc_fix_reports[name] = self.preprocessor.wbc_unit_fix_summary()
            print("— WBC 단위 보정 (≥1000 → ÷1000, K/µL)")
            self._show_report(self.wbc_fix_reports[name], empty_message="  보정 없음")

            df = self.preprocessor.fix_rbc_unit(df)
            self.rbc_fix_reports[name] = self.preprocessor.rbc_unit_fix_summary()
            print("— RBC 단위 보정 (≥100 → ÷100, M/µL)")
            self._show_report(self.rbc_fix_reports[name], empty_message="  보정 없음")

            df = self.preprocessor.fix_mch_mcv_swap(df)
            self.mch_mcv_swap_reports[name] = self.preprocessor.mch_mcv_swap_summary()
            print("— MCH·MCV 뒤바뀜 교환")
            self._show_report(
                self.mch_mcv_swap_reports[name],
                empty_message="  교환 없음",
            )

            self.processed[name] = df

    def run_lab_encoding(self) -> None:
        """정성 검사 음성/양성 인코딩."""
        for name, df in self.processed.items():
            pre_values = self.lab_encoder.capture_pre_values(df)
            self.processed[name] = self.lab_encoder.transform(df, pre_values=pre_values)
            self.lab_encoding_reports[name] = self.lab_encoder.encoding_summary()

            for field in self.lab_encoder.field_mapping_reports():
                print(f"\n— {field}")
                for outcome, line in self.lab_encoder.field_mapping_lines(field).items():
                    print(f"  {outcome}: {line}")

    def run_range_blanking(self) -> None:
        """임상 불가능 수치 공백 처리."""
        for name, df in self.processed.items():
            self.processed[name] = self.range_blanker.transform(df)
            self.blanking_reports[name] = self.range_blanker.blanking_summary()

            print(f"\n[{name}] 임상 불가능 수치 공백 처리")
            if self.blanking_reports[name].empty:
                print("  공백 처리된 인자 없음")
            else:
                self._display_wide(self.blanking_reports[name])

    def run_imputations(self) -> None:
        """임상·결측 보완."""
        for name in self.processed:
            before_impute = self.processed[name].copy()
            self.processed[name] = self.preprocessor.transform_imputations(self.processed[name])
            self.imputation_reports[name] = self.preprocessor.imputation_summary()
            self.missing_reports[name] = self.preprocessor.missing_rate_comparison(
                before_impute, self.processed[name]
            )

            print(f"\n[{name}] 임상·결측 보완")
            if not self.imputation_reports[name].empty:
                self._display(self.imputation_reports[name])

            missing_display = self.missing_reports[name].copy()
            for col in ("missing_pct_before", "missing_pct_after"):
                missing_display[col] = missing_display[col].map(lambda x: f"{x:.2f}%")
            missing_display["pct_point_change"] = missing_display["pct_point_change"].map(
                lambda x: f"{x:+.2f}%p"
            )
            print("— 결측 비율 (공백 처리 후 → 임상 보완 후)")
            self._display(missing_display)

    def run_column_filter(self) -> None:
        """고결측 컬럼 제외."""
        always_drop = sorted(self.config.extra_always_drop_columns)
        print(f"항상 제외 컬럼: {always_drop}")

        threshold_pct = self.config.missing_rate_threshold * 100
        for name, df in self.processed.items():
            cols_before = df.shape[1]
            filtered_df, report = self.column_filter.transform(df)
            self.processed[name] = filtered_df
            dropped = self.column_filter.dropped_summary(report)
            self.dropped_reports[name] = dropped

            print(f"\n[{name}] 결측 ≥ {threshold_pct:.0f}% 인자 제외")
            print(
                f"  컬럼 수: {cols_before} → {filtered_df.shape[1]} "
                f"(제거 {cols_before - filtered_df.shape[1]}개)"
            )
            if dropped.empty:
                print("  제거된 인자 없음")
            else:
                self._display(dropped[["field", "missing_count", "missing_pct"]])

    def run_export(self) -> None:
        """저장 전 DataFrame dtype 요약을 출력한 뒤 xlsx로 저장한다."""
        for name, filename in self.config.resolve_export_filenames().items():
            df = self.processed[name]
            self._print_save_preview(name, df)
            path = self.preprocessor.export(
                df, self.config.output_dir, filename
            )
            print(f"Saved: {path}  ({len(df):,}명, {df.shape[1]}컬럼)")

    def _print_save_preview(self, name: str, df: pd.DataFrame) -> None:
        """저장 직전 DataFrame의 행·열 수와 numeric/object 인자 목록을 출력한다."""
        numeric_df, object_df, other_df = self._dtype_summary_tables(df)
        numeric_fields = numeric_df["field"].tolist()
        object_fields = object_df["field"].tolist()
        other_fields = other_df["field"].tolist()

        print(f"\n[{name}] 저장 전 DataFrame")
        print(f"  행={len(df):,}, 열={df.shape[1]}")
        print(
            f"  numeric ({len(numeric_fields)}개): "
            + (", ".join(numeric_fields) if numeric_fields else "없음")
        )
        print(
            f"  object ({len(object_fields)}개): "
            + (", ".join(object_fields) if object_fields else "없음")
        )
        if other_fields:
            print(
                f"  기타 ({len(other_fields)}개): " + ", ".join(other_fields)
            )

    @staticmethod
    def _dtype_summary_tables(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """컬럼을 numeric / object / 기타 dtype으로 나눈 요약표를 반환한다."""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        known = set(numeric_cols) | set(object_cols)
        other_cols = [c for c in df.columns if c not in known]

        def _table(columns: list[str]) -> pd.DataFrame:
            if not columns:
                return pd.DataFrame(columns=["field", "dtype"])
            return pd.DataFrame(
                {"field": columns, "dtype": [str(df[c].dtype) for c in columns]}
            )

        return _table(numeric_cols), _table(object_cols), _table(other_cols)

    def _build_column_filter(self) -> MissingRateColumnFilter:
        """현재 config로 MissingRateColumnFilter를 생성한다."""
        return MissingRateColumnFilter(
            max_missing_rate=self.config.missing_rate_threshold,
            preserve_columns=self.config.resolve_preserve_columns(),
            always_drop_columns=self.config.extra_always_drop_columns,
        )

    @staticmethod
    def _display(df: pd.DataFrame) -> None:
        """노트북·터미널 공통 표 출력."""
        try:
            from IPython.display import display

            display(df)
        except ImportError:
            print(df.to_string())

    @classmethod
    def _display_wide(cls, df: pd.DataFrame) -> None:
        """긴 문자열 컬럼이 있는 표 출력."""
        try:
            from IPython.display import display

            with pd.option_context("display.max_colwidth", None, "display.width", None):
                display(df)
        except ImportError:
            print(df.to_string())

    @classmethod
    def _show_report(cls, report: pd.DataFrame, *, empty_message: str) -> None:
        """요약표가 비었을 때 메시지, 아니면 표 출력."""
        if report.empty:
            print(empty_message)
        else:
            try:
                from IPython.display import display

                with pd.option_context("display.max_colwidth", None):
                    display(report)
            except ImportError:
                print(report.to_string())
