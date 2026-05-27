import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from core import set_korean_font

set_korean_font()

LABEL_NAMES = {0: "label=0", 1: "label=1"}
DATASET_COLORS = {
    ("pre-diabetes", 0): "#4C72B0",
    ("pre-diabetes", 1): "#DD8452",
    ("diabetes", 0): "#55A868",
    ("diabetes", 1): "#C44E52",
}


class DatasetIntervalAnalyzer:
    """two_visit / multi_visit 데이터셋을 통합하여 검진 간격 분포를 분석하고 시각화한다."""

    def __init__(self, *json_paths: Path) -> None:
        """
        Args:
            *json_paths: 분석할 JSON 파일 경로들 (two_visit_dataset.json, multi_visit_dataset.json 등)
        """
        self._df = self._load_and_merge(list(json_paths))

    def _load_and_merge(self, paths: list[Path]) -> pd.DataFrame:
        """여러 JSON 파일을 로드하여 단일 DataFrame으로 병합한다."""
        frames = []
        for path in paths:
            records = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(records.values())
            source = "2회" if "two_visit" in path.name else "3회+"
            df["source"] = source
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        merged["current_checkup_date"] = pd.to_datetime(merged["current_checkup_date"])
        merged["future_checkup_date"] = pd.to_datetime(merged["future_checkup_date"])
        merged["interval_days"] = (
            merged["future_checkup_date"] - merged["current_checkup_date"]
        ).dt.days
        return merged

    def summary(self) -> pd.DataFrame:
        """dataset × label × source 별 검진 간격 기초 통계를 반환한다."""
        rows = []
        for (ds, label, src), grp in self._df.groupby(["dataset", "label", "source"]):
            ivs = grp["interval_days"].dropna()
            rows.append({
                "dataset": ds,
                "label": label,
                "source": src,
                "count": len(ivs),
                "mean_days": round(ivs.mean(), 1),
                "median_days": round(ivs.median(), 1),
                "q1_days": round(ivs.quantile(0.25), 1),
                "q3_days": round(ivs.quantile(0.75), 1),
                "min_days": int(ivs.min()),
                "max_days": int(ivs.max()),
            })
        return pd.DataFrame(rows).sort_values(["dataset", "label", "source"]).reset_index(drop=True)

    def plot_interval_distribution(
        self,
        output_dir: Path | None = None,
        show: bool = True,
    ) -> None:
        """dataset × label 조합별 검진 간격 분포를 히스토그램으로 시각화한다.

        각 패널에는 중앙값 점선과 1년·2년·3년 기준선이 함께 표시된다.

        Args:
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
        """
        datasets = ["pre-diabetes", "diabetes"]
        labels = [0, 1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
        fig.suptitle("검진 간격 분포 (dataset × label)", fontsize=14, fontweight="bold")

        for row_idx, ds in enumerate(datasets):
            for col_idx, lb in enumerate(labels):
                ax = axes[row_idx][col_idx]
                subset = (
                    self._df[
                        (self._df["dataset"] == ds) & (self._df["label"] == lb)
                    ]["interval_days"].dropna() / 365.25
                )

                color = DATASET_COLORS.get((ds, lb), "#888888")
                ax.hist(subset, bins=30, color=color, edgecolor="white", alpha=0.85)

                median_val = subset.median()
                ax.axvline(median_val, color="black", linestyle="--", linewidth=1.5,
                           label=f"중앙값: {median_val:.2f}년")

                ax.set_title(f"{ds}  /  label={lb}  (n={len(subset):,})", fontsize=10)
                ax.set_xlabel("검진 간격 (년)")
                ax.set_ylabel("건수")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                ax.legend(fontsize=8.5)

        plt.tight_layout()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "interval_dist_by_dataset_label.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved: {p}")
        if show:
            plt.show()
        else:
            plt.close(fig)
