import json
from pathlib import Path

import pandas as pd


class DataLoader:
    """JSON 형식의 건강검진 데이터를 로드하여 DataFrame으로 반환한다."""

    DATA_DIR = Path(__file__).parent.parent / "data"

    def __init__(self, filename: str, exclude_user_keys: list[str] | None = None) -> None:
        """
        Args:
            filename: data/ 디렉터리 내 JSON 파일명 (예: 'adoc_v1.total_checkups.json')
            exclude_user_keys: 분석에서 제외할 user_key 목록
        """
        self.filepath = self.DATA_DIR / filename
        self.exclude_user_keys: frozenset[str] = frozenset(exclude_user_keys or [])

    def load_streaming(self, columns: list[str]) -> pd.DataFrame:
        """ijson 스트리밍으로 대용량 JSON 파일에서 지정 컬럼만 추출하여 DataFrame으로 반환한다.

        MongoDB 확장 JSON 형식의 날짜 필드 ({"$date": "..."}) 를 자동으로 datetime으로 변환한다.
        exclude_user_keys 필터도 적용된다.

        Args:
            columns: 추출할 최상위 필드명 목록 (예: ["user_key", "checkup_date"])

        Returns:
            지정 컬럼만 포함한 DataFrame. 날짜 컬럼은 UTC datetime으로 변환됨.
        """
        import ijson

        col_set = set(columns)
        rows: list[dict] = []

        with open(self.filepath, "rb") as f:
            for record in ijson.items(f, "item"):
                if self.exclude_user_keys and record.get("user_key") in self.exclude_user_keys:
                    continue
                row: dict = {}
                for col in col_set:
                    val = record.get(col)
                    if isinstance(val, dict) and "$date" in val:
                        val = val["$date"]
                    row[col] = val
                rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        for col in columns:
            if df[col].dtype == object:
                converted = pd.to_datetime(df[col], utc=True, errors="coerce")
                if converted.notna().any():
                    df[col] = converted
        return df

    def load(self) -> pd.DataFrame:
        """JSON 파일을 읽어 DataFrame으로 반환한다.

        MongoDB 확장 JSON 형식의 날짜 필드($date)를 datetime으로 변환하고,
        exclude_user_keys에 해당하는 행을 제거한다.
        """
        with open(self.filepath, encoding="utf-8") as f:
            records = json.load(f)

        df = pd.json_normalize(records)
        df = self._convert_date_columns(df)
        df = self._drop_excluded_users(df)
        return df

    def _drop_excluded_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """exclude_user_keys에 해당하는 행을 제거한다."""
        if not self.exclude_user_keys or "user_key" not in df.columns:
            return df
        return df[~df["user_key"].isin(self.exclude_user_keys)].reset_index(drop=True)

    def _convert_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """$date 접미사가 붙은 컬럼을 datetime으로 변환하고 컬럼명을 정리한다."""
        date_cols = [c for c in df.columns if c.endswith(".$date")]
        for col in date_cols:
            clean_name = col.removesuffix(".$date")
            df[clean_name] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df = df.drop(columns=[col])
        return df


class MedicalHistoryLoader:
    """의료 이력 JSON을 로드하고 약효(drug_effect) 기준으로 처방 정보를 추출한다."""

    DATA_DIR = Path(__file__).parent.parent / "data"

    def __init__(
        self,
        filename: str = "adoc_v1.medical_histories.json",
        drug_effects: list[str] | None = None,
    ) -> None:
        """
        Args:
            filename: data/ 디렉터리 내 JSON 파일명
            drug_effects: 필터링할 drug_effect 값 목록. None이면 전체 포함.
        """
        self.filepath = self.DATA_DIR / filename
        self.drug_effects: frozenset[str] | None = (
            frozenset(drug_effects) if drug_effects is not None else None
        )

    def load_drug_lookup(
        self,
        user_keys: set[str] | frozenset[str] | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        """user_key별 처방 목록을 반환한다.

        Args:
            user_keys: 조회할 user_key 집합. None이면 전체 사용자 포함.

        Returns:
            {user_key: [{"drug_name": str, "start_date": str (YYYY-MM-DD)}, ...]}
            start_date 오름차순 정렬. drug_effects 필터 적용 시 해당 약효 약물만 포함.
        """
        with open(self.filepath, encoding="utf-8") as f:
            records = json.load(f)

        target_keys: frozenset[str] | None = (
            frozenset(user_keys) if user_keys is not None else None
        )
        lookup: dict[str, list[dict[str, str]]] = {}

        for rec in records:
            user_key = rec.get("user_key")
            if not user_key:
                continue
            if target_keys is not None and user_key not in target_keys:
                continue

            raw_date = rec.get("start_date")
            if isinstance(raw_date, dict):
                start_date_str = raw_date.get("$date", "")
            else:
                start_date_str = str(raw_date) if raw_date else ""

            try:
                start_date = pd.to_datetime(start_date_str, utc=True).strftime("%Y-%m-%d")
            except Exception:
                start_date = ""

            for med in rec.get("medication_detail_infos", []):
                effect = med.get("drug_effect", "")
                if self.drug_effects is not None and effect not in self.drug_effects:
                    continue
                drug_name = med.get("drug_name", "")
                if not drug_name:
                    continue
                if user_key not in lookup:
                    lookup[user_key] = []
                lookup[user_key].append({"drug_name": drug_name, "start_date": start_date})

        # start_date 오름차순 정렬 후 중복 제거
        for user_key in lookup:
            seen: set[tuple[str, str]] = set()
            deduped: list[dict[str, str]] = []
            for entry in sorted(lookup[user_key], key=lambda x: x["start_date"]):
                key = (entry["drug_name"], entry["start_date"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(entry)
            lookup[user_key] = deduped

        return lookup
