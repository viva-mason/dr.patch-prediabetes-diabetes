from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from core import set_korean_font

set_korean_font()


class CheckupIntervalAnalyzer:
    """환자별 검진 횟수 분포 및 재방문 주기를 분석하고 시각화한다."""

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "checkup_date",
        user_col: str = "user_key",
    ) -> None:
        """
        Args:
            df: DataLoader로 로드한 DataFrame
            date_col: 검진일 컬럼명
            user_col: 고유 사용자 식별 컬럼명
        """
        self.date_col = date_col
        self.user_col = user_col
        self._df = df[[user_col, date_col]].dropna().copy()
        self._df[date_col] = pd.to_datetime(self._df[date_col], utc=True, errors="coerce")
        self._df = self._df.dropna(subset=[date_col])
        self._df = self._df.sort_values([user_col, date_col])

        self._checkup_counts: pd.Series = self._df.groupby(user_col).size()
        self._intervals_days: pd.Series = self._compute_intervals()

    def _compute_intervals(self) -> pd.Series:
        """환자별 연속 검진 사이의 일수 차이를 반환한다."""
        diffs = (
            self._df.groupby(self.user_col)[self.date_col]
            .apply(lambda s: s.diff().dt.days.dropna())
        )
        return diffs.explode().astype(float).dropna()

    def summary(self) -> dict:
        """반복 검진 현황 기초 통계를 딕셔너리로 반환한다."""
        counts = self._checkup_counts
        intervals = self._intervals_days

        return {
            "total_patients": int(counts.shape[0]),
            "single_visit": int((counts == 1).sum()),
            "repeat_visit": int((counts >= 2).sum()),
            "repeat_ratio": round((counts >= 2).mean() * 100, 1),
            "max_visits": int(counts.max()),
            "median_interval_days": round(float(intervals.median()), 1),
            "mean_interval_days": round(float(intervals.mean()), 1),
            "interval_q1": round(float(intervals.quantile(0.25)), 1),
            "interval_q3": round(float(intervals.quantile(0.75)), 1),
        }

    def plot_analysis(self, output_dir: Path | None = None, show: bool = True) -> None:
        """검진 횟수 분포·재방문 주기 분포·누적 재방문 비율을 시각화한다.

        Args:
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
        """
        counts = self._checkup_counts
        intervals = self._intervals_days

        # 검진 횟수 구간 (1, 2, 3, 4, 5+)
        bins = counts.clip(upper=5).replace(5, 5)
        freq_labels = ["1회", "2회", "3회", "4회", "5회+"]
        freq_values = [(bins == i).sum() if i < 5 else (bins >= 5).sum() for i in range(1, 6)]

        # 재방문 주기 구간 (월 단위)
        interval_months = intervals / 30.44
        month_bins = [0, 6, 12, 18, 24, 36, 48, np.inf]
        month_labels = ["~6M", "6~12M", "12~18M", "18~24M", "24~36M", "36~48M", "48M+"]
        interval_hist = pd.cut(interval_months, bins=month_bins, labels=month_labels,
                               right=True).value_counts().reindex(month_labels)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Repeated Checkup Analysis", fontsize=14, fontweight="bold")

        # ── ① 환자별 검진 횟수 분포 ──────────────────────────
        bars = axes[0].bar(freq_labels, freq_values, color="#4C72B0", edgecolor="white")
        axes[0].set_title("Checkup Frequency per Patient")
        axes[0].set_xlabel("Number of Checkups")
        axes[0].set_ylabel("Patients")
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        axes[0].set_ylim(top=max(freq_values) * 1.15)
        for bar, val in zip(bars, freq_values):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(freq_values) * 0.02,
                         f"{val:,}", ha="center", va="bottom", fontsize=9)

        # ── ② 재방문 주기 분포 (월 단위) ─────────────────────
        bars2 = axes[1].bar(month_labels, interval_hist.values, color="#DD8452", edgecolor="white")
        axes[1].set_title("Checkup Interval Distribution")
        axes[1].set_xlabel("Interval")
        axes[1].set_ylabel("Count")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        axes[1].set_ylim(top=interval_hist.values.max() * 1.15)
        axes[1].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars2, interval_hist.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + interval_hist.values.max() * 0.02,
                         f"{val:,}", ha="center", va="bottom", fontsize=9)

        # ── ③ 검진 횟수 누적 비율 ─────────────────────────────
        sorted_counts = counts.sort_values()
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
        axes[2].plot(sorted_counts.values, cumulative, color="#55A868", linewidth=2)
        axes[2].axvline(x=2, color="#C44E52", linestyle="--", linewidth=1.2,
                        label=f"2회 이상: {(counts >= 2).mean()*100:.1f}%")
        axes[2].set_title("Cumulative Ratio by Checkup Count")
        axes[2].set_xlabel("Number of Checkups")
        axes[2].set_ylabel("Cumulative (%)")
        axes[2].set_xlim(left=1)
        axes[2].legend(fontsize=9)
        axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "repeated_checkup_analysis.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
