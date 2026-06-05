import re

with open("core/dataset_builder.py", "r") as f:
    content = f.read()

# Replace __init__ signature
old_init = """    def __init__(
        self,
        *transition_json_paths: Path,
        checkups_filename: str = "adoc_v1.total_checkups.json",
        max_interval_years: float = 3.0,
        exclude_user_keys: list[str] | None = None,
    ) -> None:
        \"\"\"
        Args:
            *transition_json_paths: two_visit / multi_visit 데이터셋 JSON 경로
            checkups_filename: data/ 디렉터리 내 원본 검진 JSON 파일명
            max_interval_years: current → future 검진 간격 상한 (년)
            exclude_user_keys: 원본 검진에서 제외할 user_key 목록
        \"\"\"
        self.transition_json_paths = list(transition_json_paths)
        self.checkups_filepath = self.DATA_DIR / checkups_filename
        self.max_interval_years = max_interval_years
        self.exclude_user_keys: frozenset[str] = frozenset(exclude_user_keys or [])
        self._datasets: dict[str, dict[str, dict]] | None = None"""

new_init = """    def __init__(
        self,
        *transition_json_paths: Path,
        checkups_filename: str = "adoc_v1.total_checkups.json",
        max_interval_years: float = 3.0,
        negative_buffer_years: float = 0.5,
        exclude_user_keys: list[str] | None = None,
        exclude_user_key_prefixes: list[str] | None = None,
    ) -> None:
        \"\"\"
        Args:
            *transition_json_paths: two_visit / multi_visit 데이터셋 JSON 경로
            checkups_filename: data/ 디렉터리 내 원본 검진 JSON 파일명
            max_interval_years: current → future 검진 간격 상한 (년).
                label=1은 이 값 이하, label=0은 이 값 초과인 레코드만 유지.
            negative_buffer_years: label=0의 간격 상한 추가 여유 (년).
                label=0은 max_interval_years 초과 AND
                max_interval_years + negative_buffer_years 이하인 레코드만 유지.
            exclude_user_keys: 원본 검진에서 제외할 user_key 정확값 목록
            exclude_user_key_prefixes: 해당 접두사로 시작하는 user_key를 제외할 접두사 목록
        \"\"\"
        self.transition_json_paths = list(transition_json_paths)
        self.checkups_filepath = self.DATA_DIR / checkups_filename
        self.max_interval_years = max_interval_years
        self.negative_buffer_years = negative_buffer_years
        self.exclude_user_keys: frozenset[str] = frozenset(exclude_user_keys or [])
        self.exclude_user_key_prefixes: tuple[str, ...] = tuple(exclude_user_key_prefixes or [])
        self._datasets: dict[str, dict[str, dict]] | None = None
        self._filter_stats: dict[str, int] = {}"""

content = content.replace(old_init, new_init)

# Replace build method
old_build = """    def build(self) -> dict[str, dict[str, dict]]:
        \"\"\"검진 간격 필터를 적용하고 원본 검진 필드를 병합한 데이터셋을 반환한다.

        Returns:
            {"pre-diabetes": {record_key: record}, "diabetes": {record_key: record}}
        \"\"\"
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
        return datasets"""

new_build = """    def build(self) -> dict[str, dict[str, dict]]:
        \"\"\"검진 간격 필터·약물 이력 필터·중복 제거를 적용하고 원본 검진 필드를 병합한 데이터셋을 반환한다.

        처리 순서:
            1. 검진 간격이 max_interval_years 초과인 레코드 제외
            2. label=0 중 current~future 기간 내 당뇨병용제 처방 이력이 있는 레코드 제외
            3. 동일 dataset 내 user_key 중복 제거
               - label=0 후보 다수: future_glucose 최저 선택
               - label=1 후보 다수: future_glucose 최고 선택
               - label=0/1 혼재: label=1 우선 보존

        Returns:
            {"pre-diabetes": {record_key: record}, "diabetes": {record_key: record}}
        \"\"\"
        from collections import defaultdict
        from datetime import timedelta
        
        transition_records = self._load_transition_records()
        checkup_index = self._build_checkup_index()
        max_interval_days = self.max_interval_years * 365.25

        ds_names = ["pre-diabetes", "diabetes"]
        candidates: dict[str, list[tuple[str, dict]]] = {ds: [] for ds in ds_names}
        per_ds: dict[str, dict[str, int]] = {
            ds: {
                "n_raw_label0": 0,
                "n_raw_label1": 0,
                "n_interval_dropped_label0": 0,
                "n_interval_upper_dropped_label0": 0,
                "n_interval_dropped_label1": 0,
                "n_checkup_missing": 0,
                "n_drug_dropped": 0,
                "n_dedup_dropped": 0,
                "n_final_label0": 0,
                "n_final_label1": 0,
            }
            for ds in ds_names
        }

        # ── Step 1: 간격 필터 ────────────────────────────────────────
        max_interval_upper_days = (
            self.max_interval_years + self.negative_buffer_years
        ) * 365.25

        for record_key, meta in transition_records.items():
            dataset_name = meta.get("dataset", "")
            if dataset_name not in per_ds:
                continue
            label = meta.get("label")
            s = per_ds[dataset_name]
            if label == 0:
                s["n_raw_label0"] += 1
            elif label == 1:
                s["n_raw_label1"] += 1

            interval_days = self._interval_days(
                meta["current_checkup_date"],
                meta["future_checkup_date"],
            )
            # label=1: 간격이 max_interval_years 이하인 레코드만 유효
            # label=0: max_interval_years 초과 AND
            #          max_interval_years + negative_buffer_years 이하인 레코드만 유효
            if label == 1 and interval_days > max_interval_days:
                s["n_interval_dropped_label1"] += 1
                continue
            if label == 0 and interval_days <= max_interval_days:
                s["n_interval_dropped_label0"] += 1
                continue
            if label == 0 and interval_days > max_interval_upper_days:
                s["n_interval_upper_dropped_label0"] += 1
                continue

            lookup_key = (meta["user_key"], meta["current_checkup_date"])
            checkup = checkup_index.get(lookup_key)
            if checkup is None:
                s["n_checkup_missing"] += 1
                continue

            record = dict(checkup)
            record.update(self._normalize_transition_meta(meta))
            record.pop("transition", None)
            record["interval_days"] = interval_days
            candidates[dataset_name].append((record_key, record))

        # ── Step 2: 당뇨병용제 이력 필터 (label=0만) ─────────────────
        filtered: dict[str, list[tuple[str, dict]]] = {}
        for ds, pairs in candidates.items():
            kept = []
            for rk, rec in pairs:
                if rec.get("label") == 0:
                    current_date = datetime.strptime(
                        rec["current_checkup_date"], "%Y-%m-%d"
                    )
                    cutoff_date = (
                        current_date
                        + timedelta(days=self.max_interval_years * 365.25)
                    ).strftime("%Y-%m-%d")
                    if self._has_drug_in_interval(rec, cutoff_date):
                        per_ds[ds]["n_drug_dropped"] += 1
                        continue
                rec.pop("diabetes_drugs", None)  # 필터 완료 후 제거
                kept.append((rk, rec))
            filtered[ds] = kept

        # ── Step 3: 중복 제거 ─────────────────────────────────────────
        datasets: dict[str, dict[str, dict]] = {}

        for ds, pairs in filtered.items():
            n_before = len(pairs)
            by_user: dict[str, list[tuple[str, dict]]] = defaultdict(list)
            for rk, rec in pairs:
                by_user[rec["user_key"]].append((rk, rec))

            final: dict[str, dict] = {}
            for user_recs in by_user.values():
                label0 = [(rk, r) for rk, r in user_recs if r.get("label") == 0]
                label1 = [(rk, r) for rk, r in user_recs if r.get("label") == 1]

                if label1:
                    chosen_rk, chosen_rec = max(
                        label1, key=lambda x: x[1].get("future_glucose") or 0
                    )
                elif label0:
                    chosen_rk, chosen_rec = min(
                        label0,
                        key=lambda x: x[1].get("future_glucose") or float("inf"),
                    )
                else:
                    continue

                final[chosen_rk] = chosen_rec

            per_ds[ds]["n_dedup_dropped"] = n_before - len(final)
            per_ds[ds]["n_final_label0"] = sum(
                1 for r in final.values() if r.get("label") == 0
            )
            per_ds[ds]["n_final_label1"] = sum(
                1 for r in final.values() if r.get("label") == 1
            )
            datasets[ds] = final

        self._filter_stats = per_ds
        self._datasets = datasets
        return datasets"""

content = content.replace(old_build, new_build)

# Replace print_label_distribution
old_print = """    def print_label_distribution(self, dataset_name: str) -> None:
        \"\"\"데이터셋의 shape과 label 0/1 분포를 출력한다.\"\"\"
        df = self.to_dataframe(dataset_name)
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        print(f"── {dataset_name} ──")
        print(f"  shape : {df.shape}")
        for label, count in counts.items():
            print(f"  label {label} : {count:5d}개  ({count/total*100:.1f}%)")
        print()"""

new_print = """    def print_label_distribution(self, dataset_name: str) -> None:
        \"\"\"데이터셋의 shape과 label 0/1 분포를 출력한다.\"\"\"
        df = self.to_dataframe(dataset_name)
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        print(f"── {dataset_name} ──")
        print(f"  shape : {df.shape}")
        for label, count in counts.items():
            print(f"  label {label} : {count:5d}개  ({count/total*100:.1f}%)")
        print()

    def print_filter_stats(self) -> None:
        \"\"\"build() 과정의 필터링 통계를 출력한다.\"\"\"
        if not self._filter_stats:
            print("필터 통계가 없습니다. build()를 먼저 실행하세요.")
            return

        ds_order = ["pre-diabetes", "diabetes"]
        W = 30
        C = 14
        sep_heavy = "═" * (W + C * 3)
        sep_light = "─" * (W + C * 3)

        def _row(label: str, keys: list[str], indent: int = 0) -> None:
            prefix = " " * indent + label
            vals = []
            total = 0
            for ds in ds_order:
                val = sum(self._filter_stats[ds].get(k, 0) for k in keys)
                vals.append(val)
                total += val
            row_str = f"{prefix:<{W}}" + "".join(f"{v:>{C-1},d}건" for v in vals) + f"{total:>{C-1},d}건"
            print(row_str)

        print(sep_heavy)
        header = f"{'':>{W}}" + "".join(f"{ds:>{C}}" for ds in ds_order) + f"{'합계':>{C}}"
        print(header)
        print(sep_heavy)

        # ── 원천 건수 ─────────────────────────────────────────────────
        _row("원천 레코드 (label=0)", ["n_raw_label0"])
        _row("원천 레코드 (label=1)", ["n_raw_label1"])
        _row("원천 합계",             ["n_raw_label0", "n_raw_label1"])

        # ── Step 1: 간격 필터 ─────────────────────────────────────────
        upper = self.max_interval_years + self.negative_buffer_years
        print(sep_light)
        print(
            f"  [Step 1] 검진 간격 필터  "
            f"(label=1: ≤{self.max_interval_years}년 / "
            f"label=0: {self.max_interval_years}~{upper}년)"
        )
        _row("label=0 간격 부족 제외",    ["n_interval_dropped_label0"],       indent=2)
        _row("label=0 간격 초과 제외",    ["n_interval_upper_dropped_label0"], indent=2)
        _row("label=1 간격 초과 제외",    ["n_interval_dropped_label1"],       indent=2)
        _row("검진 레코드 미매칭",        ["n_checkup_missing"],               indent=2)

        # ── Step 2: 약물 이력 필터 ────────────────────────────────────
        print(sep_light)
        print(f"  [Step 2] 당뇨병용제 이력 필터  (label=0, 기준일 이후 {self.max_interval_years}년 이내)")
        _row("약물 이력 제외",         ["n_drug_dropped"],            indent=2)

        # ── Step 3: 중복 제거 ─────────────────────────────────────────
        print(sep_light)
        print("  [Step 3] 중복 제거")
        _row("중복 제거 제외",         ["n_dedup_dropped"],           indent=2)

        # ── 최종 결과 ─────────────────────────────────────────────────
        print(sep_heavy)
        _row("최종 포함 (label=0)",    ["n_final_label0"])
        _row("최종 포함 (label=1)",    ["n_final_label1"])
        _row("최종 합계",             ["n_final_label0", "n_final_label1"])
        print(sep_heavy)
        print()"""

content = content.replace(old_print, new_print)

# Replace _build_checkup_index
old_index = """    def _build_checkup_index(self) -> dict[tuple[str, str], dict]:
        \"\"\"(user_key, checkup_date) → 원본 검진 레코드 인덱스를 구축한다.\"\"\"
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
        return index"""

new_index = """    def _build_checkup_index(self) -> dict[tuple[str, str], dict]:
        \"\"\"(user_key, checkup_date) → 원본 검진 레코드 인덱스를 구축한다.\"\"\"
        with open(self.checkups_filepath, encoding="utf-8") as f:
            records: list[dict] = json.load(f)

        index: dict[tuple[str, str], dict] = {}
        for record in records:
            user_key = record.get("user_key")
            if user_key in self.exclude_user_keys:
                continue
            if user_key and any(user_key.startswith(prefix) for prefix in self.exclude_user_key_prefixes):
                continue
            date_str = self._parse_date(record.get("checkup_date"))
            if user_key is None or date_str is None:
                continue
            index[(user_key, date_str)] = record
        return index"""

content = content.replace(old_index, new_index)

# Add _has_drug_in_interval if missing
if "_has_drug_in_interval" not in content:
    content += """
    @staticmethod
    def _has_drug_in_interval(record: dict, cutoff_date: str) -> bool:
        \"\"\"current_checkup_date ~ cutoff_date 기간 내 당뇨병용제 처방 여부를 반환한다.\"\"\"
        drugs = record.get("diabetes_drugs")
        if not drugs:
            return False
        current = record.get("current_checkup_date", "")
        return any(
            current <= drug.get("start_date", "") <= cutoff_date
            for drug in drugs
            if drug.get("start_date")
        )
"""

with open("core/dataset_builder.py", "w") as f:
    f.write(content)
