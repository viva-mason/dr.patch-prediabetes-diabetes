from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from core import set_korean_font

set_korean_font()

# 분석 대상 필드 기본값 (코드 → 표시명)
DEFAULT_FIELDS: dict[str, str] = {
    "CH164": "공복혈당",
    "CH161": "당화혈색소",
}


class MissingValueAnalyzer:
    """반복 수검자(2회 이상)를 대상으로 특정 검진 항목의 결측 현황을 분석한다."""

    def __init__(
        self,
        df: pd.DataFrame,
        fields: dict[str, str] | None = None,
        date_col: str = "checkup_date",
        user_col: str = "user_key",
        detail_col: str = "detail_infos",
        min_visits: int = 2,
    ) -> None:
        """
        Args:
            df: DataLoader로 로드한 DataFrame
            fields: 분석할 {검진코드: 표시명} 딕셔너리. None이면 DEFAULT_FIELDS 사용
            date_col: 검진일 컬럼명
            user_col: 고유 사용자 식별 컬럼명
            detail_col: 세부 검진 항목 컬럼명
            min_visits: 반복 수검 기준 최소 방문 횟수 (기본 2)
        """
        self.fields = fields if fields is not None else DEFAULT_FIELDS
        self.user_col = user_col
        self.detail_col = detail_col
        self.min_visits = min_visits

        visit_counts = df.groupby(user_col).size()
        repeat_users = visit_counts[visit_counts >= min_visits].index
        self._repeat_df = df[df[user_col].isin(repeat_users)].reset_index(drop=True).copy()
        self._repeat_df["_rid"] = self._repeat_df.index

        self._detail_norm = self._build_detail_norm()

    def _build_detail_norm(self) -> pd.DataFrame:
        """detail_infos를 explode·정규화하여 반환한다."""
        exploded = self._repeat_df[["_rid", self.detail_col]].explode(self.detail_col)
        valid = exploded[exploded[self.detail_col].notna()]
        detail = pd.json_normalize(valid[self.detail_col].tolist())
        detail["_rid"] = valid["_rid"].values
        return detail

    def summary(self) -> pd.DataFrame:
        """항목별 결측 현황을 DataFrame으로 반환한다.

        Returns:
            columns: field_code, field_name, total_records,
                     present_records, missing_records, missing_rate_pct
        """
        total = len(self._repeat_df)
        rows = []
        for code, name in self.fields.items():
            present = self._detail_norm[
                self._detail_norm["small_checkup_code"] == code
            ]["_rid"].nunique()
            missing = total - present
            rows.append({
                "field_code": code,
                "field_name": name,
                "total_records": total,
                "present_records": present,
                "missing_records": missing,
                "missing_rate_pct": round(missing / total * 100, 1),
            })
        return pd.DataFrame(rows)

    def plot_missing(
        self,
        output_dir: Path | None = None,
        show: bool = True,
        filename: str = "missing_value_analysis.png",
    ) -> None:
        """항목별 결측/측정 비율을 수평 누적 막대 차트로 시각화한다.

        Args:
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
            filename: 저장할 파일명 (기본값: "missing_value_analysis.png").
        """
        fig, ax = plt.subplots(figsize=(9, max(3, len(self.fields) * 1.2)))
        fig.suptitle(
            f"결측치 현황 (반복 수검자 {len(self._repeat_df):,}건 기준)",
            fontsize=13, fontweight="bold",
        )
        self._draw_missing_bars(ax, self.summary())
        plt.tight_layout()
        self._save_or_show(fig, output_dir, filename, show)

    @classmethod
    def plot_missing_comparison(
        cls,
        sources: list[tuple[str, "MissingValueAnalyzer"]],
        output_dir: Path | None = None,
        show: bool = True,
        filename: str = "missing_value_analysis.png",
    ) -> None:
        """여러 데이터소스의 결측 현황을 하나의 그림에 나란히 시각화한다.

        Args:
            sources: [(레이블, MissingValueAnalyzer), ...] 리스트.
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
            filename: 저장할 파일명 (기본값: "missing_value_analysis.png").
        """
        n = len(sources)
        row_heights = [max(2, len(analyzer.fields) * 1.2) for _, analyzer in sources]
        fig, axes = plt.subplots(
            n, 1,
            figsize=(9, sum(row_heights) + 0.8),
            gridspec_kw={"height_ratios": row_heights},
        )
        if n == 1:
            axes = [axes]

        fig.suptitle("결측치 현황 비교", fontsize=13, fontweight="bold")

        for ax, (label, analyzer) in zip(axes, sources):
            ax.set_title(
                f"{label}  (반복 수검자 {len(analyzer._repeat_df):,}건 기준)",
                fontsize=11,
            )
            analyzer._draw_missing_bars(ax, analyzer.summary())

        plt.tight_layout()
        cls._save_or_show(fig, output_dir, filename, show)

    # ── internal helpers ───────────────────────────────────────────────────

    def _draw_missing_bars(self, ax: plt.Axes, summary: pd.DataFrame) -> None:
        """수평 누적 막대 차트를 ax에 그린다."""
        labels = summary["field_name"].tolist()
        present_pct = (summary["present_records"] / summary["total_records"] * 100).tolist()
        missing_pct = summary["missing_rate_pct"].tolist()

        y = range(len(labels))
        bars_p = ax.barh(list(y), present_pct, color="#4C72B0", label="측정")
        bars_m = ax.barh(list(y), missing_pct, left=present_pct, color="#C44E52", label="결측")

        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("비율 (%)")
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(loc="lower right")

        for bar_p, bar_m, row in zip(bars_p, bars_m, summary.itertuples()):
            ax.text(
                bar_p.get_width() / 2, bar_p.get_y() + bar_p.get_height() / 2,
                f"{row.present_records:,}건\n({100 - row.missing_rate_pct:.1f}%)",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            )
            if row.missing_rate_pct > 5:
                ax.text(
                    bar_p.get_width() + bar_m.get_width() / 2,
                    bar_m.get_y() + bar_m.get_height() / 2,
                    f"{row.missing_records:,}건\n({row.missing_rate_pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                )

    @staticmethod
    def _save_or_show(
        fig: plt.Figure,
        output_dir: Path | None,
        filename: str,
        show: bool,
    ) -> None:
        """그림을 저장하거나 화면에 표시한다."""
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / filename
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
