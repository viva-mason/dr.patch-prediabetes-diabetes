from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from core import set_korean_font

set_korean_font()


class CheckupDateAnalyzer:
    """checkup_date 컬럼의 분포를 분석하고 시각화한다."""

    MAX_VALID_YEAR: int = 2100

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "checkup_date",
        user_col: str = "user_key",
    ) -> None:
        """
        Args:
            df: DataLoader로 로드한 DataFrame
            date_col: 분석 대상 날짜 컬럼명
            user_col: 고유 사용자 식별 컬럼명
        """
        self.df = df.copy()
        self.date_col = date_col
        self.user_col = user_col
        raw = self.df[date_col].dropna()
        self._dates: pd.Series = raw[raw.dt.year <= self.MAX_VALID_YEAR]

    def summary(self) -> dict:
        """날짜 범위·건수·고유 사용자 수 등 기초 통계를 딕셔너리로 반환한다."""
        valid_idx = self._dates.index
        user_count = (
            int(self.df.loc[valid_idx, self.user_col].nunique())
            if self.user_col in self.df.columns
            else None
        )
        return {
            "count": int(self._dates.count()),
            "user_count": user_count,
            "min": self._dates.min(),
            "max": self._dates.max(),
            "unique_years": sorted(self._dates.dt.year.unique().tolist()),
        }

    def plot_distribution(self, output_dir: Path | None = None, show: bool = True) -> None:
        """연도·월별 검진 건수 분포를 시각화한다.

        상단 stats 패널에 총 검진 건수·인원·기간을 표시하고,
        두 차트는 동일한 datetime x축을 공유한다(sharex).

        Args:
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
        """
        import matplotlib.dates as mdates
        import matplotlib.gridspec as gridspec

        s = self.summary()
        dates = self._dates.dt.tz_convert("Asia/Seoul")
        valid_df = self.df.loc[self._dates.index].copy()
        valid_df["_year"] = dates.dt.year.values

        year_counts = dates.dt.year.value_counts().sort_index()
        year_index = pd.to_datetime([f"{y}-01-01" for y in year_counts.index])

        year_users = (
            valid_df.groupby("_year")[self.user_col].nunique().sort_index()
            if self.user_col in valid_df.columns
            else None
        )

        month_counts = dates.dt.to_period("M").value_counts().sort_index()
        month_index = month_counts.index.to_timestamp()

        fig = plt.figure(figsize=(12, 11))
        fig.suptitle("Total Checkup Date Distribution", fontsize=15, fontweight="bold", y=0.99)

        gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.6)
        ax_users = fig.add_subplot(gs[0])
        ax_annual = fig.add_subplot(gs[1], sharex=ax_users)
        ax_monthly = fig.add_subplot(gs[2], sharex=ax_users)

        # ── 연도별 고유 사용자 수 막대 그래프 ─────────────────
        if year_users is not None:
            user_year_index = pd.to_datetime([f"{y}-01-01" for y in year_users.index])
            ax_users.bar(user_year_index, year_users.values, width=330,
                         color="#55A868", edgecolor="white", align="center")
            ax_users.set_title("Annual Unique User Count")
            ax_users.set_ylabel("Users")
            ax_users.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax_users.xaxis.set_major_locator(mdates.YearLocator())
            ax_users.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_users.tick_params(axis="x", labelbottom=True, rotation=45)
            ax_users.set_ylim(top=year_users.values.max() * 1.15)
            for x, val in zip(user_year_index, year_users.values):
                ax_users.text(x, val + year_users.values.max() * 0.02,
                              f"{val:,}", ha="center", va="bottom", fontsize=9)

        # ── 연도별 검진 건수 막대 그래프 ──────────────────────
        ax_annual.bar(year_index, year_counts.values, width=330,
                      color="#4C72B0", edgecolor="white", align="center")
        ax_annual.set_title("Annual Checkup Count")
        ax_annual.set_ylabel("Count")
        ax_annual.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_annual.xaxis.set_major_locator(mdates.YearLocator())
        ax_annual.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_annual.tick_params(axis="x", labelbottom=True, rotation=45)
        ax_annual.set_ylim(top=year_counts.values.max() * 1.15)
        for x, val in zip(year_index, year_counts.values):
            ax_annual.text(x, val + year_counts.values.max() * 0.02,
                           f"{val:,}", ha="center", va="bottom", fontsize=9)

        # ── 월별 선 그래프 ────────────────────────────────────
        ax_monthly.plot(month_index, month_counts.values, marker="o", markersize=3,
                        linewidth=1.5, color="#DD8452")
        ax_monthly.fill_between(month_index, month_counts.values, alpha=0.2, color="#DD8452")
        ax_monthly.set_title("Monthly Checkup Count")
        ax_monthly.set_xlabel("Year")
        ax_monthly.set_ylabel("Count")
        ax_monthly.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_monthly.xaxis.set_major_locator(mdates.YearLocator())
        ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_monthly.tick_params(axis="x", rotation=45)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "checkup_date_distribution.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
