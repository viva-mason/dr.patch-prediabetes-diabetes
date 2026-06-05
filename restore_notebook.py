import json

with open("notebooks/260526_create_dataset.ipynb", "r") as f:
    nb = json.load(f)

# Cell 1
nb["cells"][1]["source"] = [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.insert(0, str(Path(\"..\").resolve()))\n",
    "\n",
    "from core.dataset_builder import TransitionDatasetBuilder\n",
    "\n",
    "OUTPUT_DIR = Path(\"../outputs/260526_create_dataset\")\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "EDA_OUTPUT_DIR = Path(\"../outputs/260522_EDA\")\n",
    "\n",
    "# current_checkup_date → future_checkup_date 간격 상한 (년)\n",
    "# label=1: 이 값 이하인 레코드만 유지\n",
    "MAX_INTERVAL_YEARS = 1.0\n",
    "\n",
    "# label=0의 간격 상한 여유 (년)\n",
    "# label=0: MAX_INTERVAL_YEARS 초과 AND MAX_INTERVAL_YEARS + NEGATIVE_BUFFER_YEARS 이하인 레코드만 유지\n",
    "NEGATIVE_BUFFER_YEARS = 0.5\n",
    "\n",
    "EXCLUDE_USER_KEYS = [\n",
    "    \"INVALID_RESULT\",\n",
    "]\n",
    "\n",
    "# 해당 접두사로 시작하는 user_key를 가진 레코드 제외 (익명·워크인 수검자)\n",
    "EXCLUDE_USER_KEY_PREFIXES = [\n",
    "    \"ANONYMOUS\",\n",
    "    \"WALKIN\",\n",
    "]"
]

# Cell 2
nb["cells"][2]["source"] = [
    "## 1. 데이터셋 생성\n",
    "\n",
    "`two_visit_dataset.json` (2회 수검) + `multi_visit_dataset.json` (3회+ 수검)을 통합한 뒤,  \n",
    "아래 필터를 순서대로 적용하여 최종 학습용 데이터셋을 생성합니다.\n",
    "\n",
    "### 필터링 규칙\n",
    "\n",
    "**Step 0. 익명·워크인 수검자 제외**\n",
    "- `user_key`가 `ANONYMOUS` 또는 `WALKIN`으로 시작하는 레코드 제외\n",
    "\n",
    "**Step 1. 검진 간격 필터** (label 기준 반전 적용)\n",
    "- **label=1**: 간격이 `MAX_INTERVAL_YEARS` **이하**인 레코드만 유지\n",
    "- **label=0**: 간격이 `MAX_INTERVAL_YEARS` **초과** AND `MAX_INTERVAL_YEARS + NEGATIVE_BUFFER_YEARS` **이하**인 레코드만 유지\n",
    "  - 충분한 추적 기간 동안 전이가 없었음을 확인하되, 관측 윈도우를 제한하여 label=1과 시간 범위를 맞춤\n",
    "\n",
    "**Step 2. 당뇨병용제 처방 이력 필터** (label=0만 적용)\n",
    "- label=0 레코드 중, `current_checkup_date` 이후 `MAX_INTERVAL_YEARS` 이내에 당뇨병용제 처방 이력이 있는 경우 제외\n",
    "  - 가까운 미래에 당뇨 진행 가능성이 있어 음성 레이블로 보기 어렵기 때문\n",
    "\n",
    "**Step 3. 중복 제거** (동일 dataset 내 user_key는 반드시 유일)\n",
    "- 동일 user_key에 유효한 전이 흐름이 여러 개인 경우\n",
    "  - label=0 후보가 여럿: 이후 검진일의 공복혈당이 **가장 낮은** 검진일 선택\n",
    "  - label=1 후보가 여럿: 이후 검진일의 공복혈당이 **가장 높은** 검진일 선택\n",
    "- label=0 과 label=1 후보가 동시에 존재하는 경우: **label=1 우선 보존**\n",
    "\n",
    "---\n",
    "\n",
    "결과는 `detail_infos`를 `small_checkup_name` → `value` 형태로 펼쳐  \n",
    "`pre_diabetes_dataset.xlsx`, `diabetes_dataset.xlsx`로 저장합니다."
]

# Cell 3
nb["cells"][3]["source"] = [
    "builder = TransitionDatasetBuilder(\n",
    "    EDA_OUTPUT_DIR / \"two_visit_dataset.json\",\n",
    "    EDA_OUTPUT_DIR / \"multi_visit_dataset.json\",\n",
    "    max_interval_years=MAX_INTERVAL_YEARS,\n",
    "    negative_buffer_years=NEGATIVE_BUFFER_YEARS,\n",
    "    exclude_user_keys=EXCLUDE_USER_KEYS,\n",
    "    exclude_user_key_prefixes=EXCLUDE_USER_KEY_PREFIXES,\n",
    ")\n",
    "\n",
    "datasets = builder.build()\n",
    "saved_paths = builder.export(OUTPUT_DIR)"
]

with open("notebooks/260526_create_dataset.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
