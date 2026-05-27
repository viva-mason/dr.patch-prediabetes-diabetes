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
