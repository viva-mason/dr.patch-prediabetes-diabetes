import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd


class TransitionDatasetBuilder:
    """전이 라벨 JSON과 원본 검진 JSON을 결합하여 학습용 데이터셋을 생성한다."""

    DATA_DIR = Path(__file__).parent.parent / "data"
    _EXCLUDED_EXPORT_FIELDS = frozenset({
        "record_key",
        "dataset",
        "source",
        "_id",
        "created_at",
        "updated_at",
        "order_key",
        "center_code",
        "center_name",
        "target_name",
        "checkup_type",
    })
    _TRANSITION_FIELDS = (
        "user_key",
        "current_checkup_date",
        "future_checkup_date",
        "label",
        "selected_transition",
        "full_transition",
        "interval_days",
    )
    _TOP_LEVEL_CHECKUP_FIELDS = (
        "checkup_date",
        "gender",
        "birthday",
    )

    def __init__(
        self,
        *transition_json_paths: Path,
        checkups_filename: str = "adoc_v1.total_checkups.json",
        max_interval_years: float = 3.0,
        exclude_user_keys: list[str] | None = None,
    ) -> None:
        """
        Args:
            *transition_json_paths: two_visit / multi_visit 데이터셋 JSON 경로
            checkups_filename: data/ 디렉터리 내 원본 검진 JSON 파일명
            max_interval_years: current → future 검진 간격 상한 (년)
            exclude_user_keys: 원본 검진에서 제외할 user_key 목록
        """
        self.transition_json_paths = list(transition_json_paths)
        self.checkups_filepath = self.DATA_DIR / checkups_filename
        self.max_interval_years = max_interval_years
        self.exclude_user_keys: frozenset[str] = frozenset(exclude_user_keys or [])
        self._datasets: dict[str, dict[str, dict]] | None = None

    def build(self) -> dict[str, dict[str, dict]]:
        """검진 간격 필터를 적용하고 원본 검진 필드를 병합한 데이터셋을 반환한다.

        Returns:
            {"pre-diabetes": {record_key: record}, "diabetes": {record_key: record}}
        """
        transition_records = self._load_transition_records()
        checkup_index = self._build_checkup_index()
        max_interval_days = self.max_interval_years * 365.25

        datasets: dict[str, dict[str, dict]] = {
            "pre-diabetes": {},
            "diabetes": {},
        }

        for record_key, meta in transition_records.items():
            interval_days = self._interval_days(
                meta["current_checkup_date"],
                meta["future_checkup_date"],
            )
            if interval_days > max_interval_days:
                continue

            lookup_key = (meta["user_key"], meta["current_checkup_date"])
            checkup = checkup_index.get(lookup_key)
            if checkup is None:
                continue

            dataset_name = meta["dataset"]
            record = dict(checkup)
            record.update(self._normalize_transition_meta(meta))
            record.pop("transition", None)
            record["interval_days"] = interval_days
            datasets[dataset_name][record_key] = record

        self._datasets = datasets
        return datasets

    def to_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """데이터셋을 xlsx 저장용 평탄화 DataFrame으로 변환한다."""
        if self._datasets is None:
            raise RuntimeError("build()를 먼저 호출해야 to_dataframe()을 사용할 수 있습니다.")

        records = self._datasets[dataset_name]
        rows = [
            self._flatten_record(record)
            for record in records.values()
        ]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        ordered_cols = [c for c in self._TRANSITION_FIELDS if c in df.columns]
        ordered_cols.extend(c for c in self._TOP_LEVEL_CHECKUP_FIELDS if c in df.columns)
        ordered_cols.extend(
            sorted(c for c in df.columns if c not in ordered_cols)
        )
        return df[ordered_cols]

    def summary(self) -> pd.DataFrame:
        """dataset × label × source 별 건수 요약을 반환한다."""
        if self._datasets is None:
            raise RuntimeError("build()를 먼저 호출해야 summary()를 사용할 수 있습니다.")

        rows = []
        for dataset_name, records in self._datasets.items():
            for record in records.values():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "label": record["label"],
                        "source": record.get("source", ""),
                        "interval_days": record["interval_days"],
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["dataset", "label", "source", "count", "mean_interval_days"]
            )

        df = pd.DataFrame(rows)
        return (
            df.groupby(["dataset", "label", "source"], as_index=False)
            .agg(count=("label", "size"), mean_interval_days=("interval_days", "mean"))
            .sort_values(["dataset", "label", "source"])
            .reset_index(drop=True)
        )

    def export(
        self,
        output_dir: Path,
        file_format: Literal["xlsx", "json"] = "xlsx",
    ) -> dict[str, Path]:
        """pre-diabetes / diabetes 데이터셋을 파일로 저장한다.

        Args:
            output_dir: 저장 디렉터리
            file_format: 저장 형식 (`xlsx` 또는 `json`)

        Returns:
            dataset 이름 → 저장 경로 매핑
        """
        if self._datasets is None:
            self.build()

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}

        for dataset_name, records in self._datasets.items():
            stem = f"{dataset_name.replace('-', '_')}_dataset"
            if file_format == "xlsx":
                output_path = output_dir / f"{stem}.xlsx"
                self.to_dataframe(dataset_name).to_excel(
                    output_path, index=False, engine="openpyxl"
                )
            else:
                output_path = output_dir / f"{stem}.json"
                output_path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            saved_paths[dataset_name] = output_path
            print(f"Saved: {output_path}  ({len(records):,}명)")

        return saved_paths

    def _flatten_record(self, record: dict) -> dict:
        """중첩 JSON 레코드를 xlsx 행(dict)으로 평탄화한다."""
        row: dict[str, object] = {}

        for field in self._TRANSITION_FIELDS:
            if field in record:
                row[field] = record[field]

        for field in self._TOP_LEVEL_CHECKUP_FIELDS:
            if field in record:
                row[field] = self._serialize_cell_value(record[field])

        for key, value in record.items():
            if key in row or key == "detail_infos" or key in self._EXCLUDED_EXPORT_FIELDS:
                continue
            if key == "transition":
                continue
            if key in self._TRANSITION_FIELDS or key in self._TOP_LEVEL_CHECKUP_FIELDS:
                continue
            row[key] = self._serialize_cell_value(value)

        for item in record.get("detail_infos", []):
            if not isinstance(item, dict):
                continue
            name = item.get("small_checkup_name")
            if not name:
                continue
            row[name] = self._serialize_cell_value(item.get("value"))

        return row

    @staticmethod
    def _normalize_transition_meta(meta: dict) -> dict:
        """two_visit의 transition을 selected/full로 채우고 transition 필드는 제거한다."""
        normalized = {k: v for k, v in meta.items() if k != "transition"}
        transition = meta.get("transition")
        if transition is not None and str(transition).strip():
            if not normalized.get("selected_transition"):
                normalized["selected_transition"] = transition
            if not normalized.get("full_transition"):
                normalized["full_transition"] = transition
        return normalized

    def _load_transition_records(self) -> dict[str, dict]:
        """two_visit / multi_visit JSON을 단일 dict로 병합한다."""
        merged: dict[str, dict] = {}
        for path in self.transition_json_paths:
            records = json.loads(path.read_text(encoding="utf-8"))
            source = "2회" if "two_visit" in path.name else "3회+"
            for record_key, record in records.items():
                merged[record_key] = {**record, "source": source}
        return merged

    def _build_checkup_index(self) -> dict[tuple[str, str], dict]:
        """(user_key, checkup_date) → 원본 검진 레코드 인덱스를 구축한다."""
        with open(self.checkups_filepath, encoding="utf-8") as f:
            records: list[dict] = json.load(f)

        index: dict[tuple[str, str], dict] = {}
        for record in records:
            user_key = record.get("user_key")
            if user_key in self.exclude_user_keys:
                continue
            date_str = self._parse_date(record.get("checkup_date"))
            if user_key is None or date_str is None:
                continue
            index[(user_key, date_str)] = record
        return index

    @classmethod
    def _serialize_cell_value(cls, value: object) -> object:
        """중첩 JSON 값을 xlsx 셀에 저장 가능한 형태로 변환한다."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            if "$oid" in value:
                return value["$oid"]
            if "$date" in value:
                return cls._parse_date(value)
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def _parse_date(value: object) -> str | None:
        """MongoDB 확장 JSON 날짜 필드를 YYYY-MM-DD 문자열로 변환한다."""
        if isinstance(value, dict) and "$date" in value:
            raw = value["$date"]
            if isinstance(raw, str):
                return raw[:10]
            if isinstance(raw, dict) and "$numberLong" in raw:
                ms = int(raw["$numberLong"])
                return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
        if value is None:
            return None
        return str(value)[:10]

    @staticmethod
    def _interval_days(current_date: str, future_date: str) -> int:
        """current_checkup_date → future_checkup_date 사이 일수를 반환한다."""
        current = datetime.strptime(current_date, "%Y-%m-%d")
        future = datetime.strptime(future_date, "%Y-%m-%d")
        return (future - current).days
