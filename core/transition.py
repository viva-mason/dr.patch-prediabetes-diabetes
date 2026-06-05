from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from core import set_korean_font

set_korean_font()

# 공복혈당 기준 상태 레이블
STATE_NORMAL = "정상"
STATE_PRE = "당뇨병전단계"
STATE_DM = "당뇨병"
STATES = [STATE_NORMAL, STATE_PRE, STATE_DM]

STATE_COLORS = {
    STATE_NORMAL: "#4C72B0",
    STATE_PRE: "#DD8452",
    STATE_DM: "#C44E52",
}

# 공복혈당 기준값
GLUCOSE_CODE = "CH164"
NORMAL_MAX = 100       # 미만: 정상
PREDIAB_MAX = 125      # 이하: 당뇨병전단계 / 초과: 당뇨병


class GlucoseTransitionAnalyzer:
    """공복혈당 기준 당뇨 상태 전이 흐름을 분석하고 시각화한다.

    상태 기준 (공복혈당 mg/dL):
        - 정상        : < 100
        - 당뇨병전단계 : 100 이상 125 이하
        - 당뇨병       : > 125
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "checkup_date",
        user_col: str = "user_key",
        detail_col: str = "detail_infos",
        min_visits: int = 2,
    ) -> None:
        """
        Args:
            df: DataLoader로 로드한 DataFrame
            date_col: 검진일 컬럼명
            user_col: 고유 사용자 식별 컬럼명
            detail_col: 세부 검진 항목 컬럼명
            min_visits: 반복 수검 기준 최소 방문 횟수 (기본 2)
        """
        self.date_col = date_col
        self.user_col = user_col
        self.min_visits = min_visits

        visit_counts = df.groupby(user_col).size()
        repeat_users = visit_counts[visit_counts >= min_visits].index
        repeat_df = df[df[user_col].isin(repeat_users)].copy()

        self._labeled = self._build_labeled(repeat_df, detail_col)
        self._transitions = self._build_transitions()
        self._matrix = self._build_matrix()

    @classmethod
    def _extract_glucose_labeled(
        cls,
        df: pd.DataFrame,
        user_col: str,
        date_col: str,
        detail_col: str,
        source: str = "",
    ) -> pd.DataFrame:
        """DataFrame에서 공복혈당 레코드를 추출하여 state·glucose (·source) 컬럼을 부착한다.

        Args:
            df: 검진 DataFrame
            user_col: 사용자 식별 컬럼명
            date_col: 검진일 컬럼명
            detail_col: 세부 검진 항목 컬럼명
            source: 출처 레이블. 비어 있으면 source 컬럼을 추가하지 않는다.

        Returns:
            user_col, date_col, glucose, state (, source) 컬럼을 포함한 DataFrame
        """
        df = df.reset_index(drop=True).copy()
        df["_rid"] = df.index

        exploded = df[["_rid", user_col, date_col, detail_col]].explode(detail_col)
        valid = exploded[exploded[detail_col].notna()]
        if valid.empty:
            cols = [user_col, date_col, "glucose", "state"]
            if source:
                cols.append("source")
            return pd.DataFrame(columns=cols)

        detail = pd.json_normalize(valid[detail_col].tolist())
        detail["_rid"] = valid["_rid"].values
        detail[user_col] = valid[user_col].values
        detail[date_col] = valid[date_col].values

        glucose = detail[detail["small_checkup_code"] == GLUCOSE_CODE][
            ["_rid", user_col, date_col, "value"]
        ].copy()
        glucose["glucose"] = pd.to_numeric(glucose["value"], errors="coerce")
        glucose = glucose.dropna(subset=["glucose"])
        glucose["state"] = glucose["glucose"].apply(cls._label_state)
        if source:
            glucose["source"] = source
        return glucose.sort_values([user_col, date_col]).reset_index(drop=True)

    def _build_labeled(self, df: pd.DataFrame, detail_col: str) -> pd.DataFrame:
        """각 검진 레코드에 공복혈당 값과 상태 레이블을 부착한다."""
        return self._extract_glucose_labeled(df, self.user_col, self.date_col, detail_col)

    @staticmethod
    def _label_state(glucose: float) -> str:
        if glucose < NORMAL_MAX:
            return STATE_NORMAL
        elif glucose <= PREDIAB_MAX:
            return STATE_PRE
        else:
            return STATE_DM

    def _build_transitions(self) -> pd.DataFrame:
        """환자별 연속 검진 간 상태 전이 쌍을 생성한다."""
        rows = []
        for user, group in self._labeled.groupby(self.user_col):
            states = group["state"].tolist()
            dates = group[self.date_col].tolist()
            for i in range(len(states) - 1):
                rows.append({
                    self.user_col: user,
                    "from_state": states[i],
                    "to_state": states[i + 1],
                    "from_date": dates[i],
                    "to_date": dates[i + 1],
                })
        return pd.DataFrame(rows)

    def _build_matrix(self) -> pd.DataFrame:
        """전이 횟수 행렬 (from × to)을 반환한다."""
        matrix = (
            self._transitions.groupby(["from_state", "to_state"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=STATES, columns=STATES, fill_value=0)
        )
        return matrix

    @property
    def user_keys(self) -> frozenset[str]:
        """분석 대상 user_key 집합을 반환한다."""
        return frozenset(self._labeled[self.user_col].unique())

    def summary(self) -> pd.DataFrame:
        """전이 유형별 건수·비율을 DataFrame으로 반환한다."""
        counts = (
            self._transitions.groupby(["from_state", "to_state"])
            .size()
            .reset_index(name="count")
        )
        counts["label"] = counts["from_state"] + " → " + counts["to_state"]
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
        return counts.sort_values("count", ascending=False).reset_index(drop=True)

    def sequence_summary(self, n_visits: list[int] | None = None) -> dict[int, pd.DataFrame]:
        """방문 횟수별 전체 상태 시퀀스 패턴을 반환한다.

        Args:
            n_visits: 분석할 방문 횟수 목록. None이면 데이터에 존재하는 모든 횟수.

        Returns:
            {방문 횟수: DataFrame(sequence, count, pct, user_count)} 딕셔너리
        """
        visit_counts = self._labeled.groupby(self.user_col).size()
        if n_visits is None:
            n_visits = sorted(visit_counts.unique().tolist())

        result: dict[int, pd.DataFrame] = {}
        for n in n_visits:
            users_n = visit_counts[visit_counts == n].index
            if len(users_n) == 0:
                continue
            subset = self._labeled[self._labeled[self.user_col].isin(users_n)]
            sequences = subset.groupby(self.user_col)["state"].apply(
                lambda s: " → ".join(s.tolist())
            )
            counts_s = sequences.value_counts().reset_index()
            counts_s.columns = ["sequence", "count"]
            counts_s["pct"] = (counts_s["count"] / counts_s["count"].sum() * 100).round(1)
            result[n] = counts_s
        return result

    def plot_sequences(
        self,
        n_visits: list[int] | None = None,
        top_n: int = 10,
        output_dir: Path | None = None,
        show: bool = True,
    ) -> None:
        """방문 횟수별 상위 시퀀스 패턴을 수평 막대 차트로 시각화한다.

        패널이 3개 이하면 1행으로, 4개 이상이면 2열 그리드로 배치한다.
        패널마다 별도 파일로도 저장한다.

        Args:
            n_visits: 표시할 방문 횟수 목록. None이면 데이터에 존재하는 모든 횟수.
            top_n: 각 방문 횟수에서 표시할 상위 패턴 수.
            output_dir: 저장 경로.
            show: True이면 plt.show()를 호출한다.
        """
        import math

        seq_dict = self.sequence_summary(n_visits)
        if not seq_dict:
            return

        visit_counts_all = self._labeled.groupby(self.user_col).size()
        legend_patches = [mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()]

        n_panels = len(seq_dict)
        n_cols = 2 if n_panels >= 4 else n_panels
        n_rows = math.ceil(n_panels / n_cols)
        fig_w = 9 * n_cols
        fig_h = max(5, top_n * 0.55) * n_rows

        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        fig.suptitle("방문 횟수별 상태 시퀀스 패턴 (Top 10)", fontsize=14, fontweight="bold")

        # axes를 항상 1차원 리스트로 다룬다
        if n_rows == 1 and n_cols == 1:
            axes_flat = [axes_grid]
        elif n_rows == 1 or n_cols == 1:
            axes_flat = list(axes_grid.flatten())
        else:
            axes_flat = list(axes_grid.flatten())

        for ax, (n, df) in zip(axes_flat, seq_dict.items()):
            total_users = int((visit_counts_all == n).sum())
            top = df.head(top_n)

            bar_colors = [
                STATE_COLORS.get(seq.split(" → ")[0], "#888888")
                for seq in top["sequence"]
            ]
            bars = ax.barh(
                top["sequence"].tolist()[::-1],
                top["count"].tolist()[::-1],
                color=bar_colors[::-1],
                edgecolor="white",
            )
            ax.set_title(f"{n}회 수검자 ({total_users:,}명)", fontsize=11)
            ax.set_xlabel("명수")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.set_xlim(right=top["count"].max() * 1.35)

            for bar, (_, row) in zip(bars, top.iloc[::-1].iterrows()):
                ax.text(
                    bar.get_width() + top["count"].max() * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['count']:,}명 ({row['pct']}%)",
                    va="center", fontsize=8,
                )

        # 빈 패널 숨기기
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "glucose_sequences.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def export_dataset(
        self,
        output_path: Path | None = None,
        drug_lookup: dict[str, list[dict[str, str]]] | None = None,
    ) -> pd.DataFrame:
        """2회 수검자 중 첫 상태가 '정상' 또는 '당뇨병전단계'인 수검자를 분류하여 반환하고 JSON으로 저장한다.

        분류 기준:
            - "pre-diabetes" : 첫 상태 == 정상
                label=0 : 정상 → 정상
                label=1 : 정상 → 당뇨병전단계  또는  정상 → 당뇨병
            - "diabetes"     : 첫 상태 == 당뇨병전단계
                label=0 : 당뇨병전단계 → 당뇨병전단계  또는  당뇨병전단계 → 정상
                label=1 : 당뇨병전단계 → 당뇨병

        Args:
            output_path: 저장할 JSON 파일 경로. None이면 저장하지 않는다.
            drug_lookup: MedicalHistoryLoader.load_drug_lookup()의 반환값.
                None이면 약물 정보를 포함하지 않는다.

        Returns:
            columns: user_key, current_checkup_date, future_checkup_date, dataset,
                     label, selected_transition, full_transition,
                     current_glucose, future_glucose, glucose_change
        """
        visit_counts = self._labeled.groupby(self.user_col).size()
        two_visit_users = visit_counts[visit_counts == 2].index
        subset = self._labeled[self._labeled[self.user_col].isin(two_visit_users)]

        rows = []
        for user, group in subset.groupby(self.user_col):
            group = group.sort_values(self.date_col).dropna(subset=[self.date_col])
            if len(group) != 2:
                continue
            first, second = group.iloc[0], group.iloc[1]
            if pd.isna(first[self.date_col]) or pd.isna(second[self.date_col]):
                continue

            dataset: str | None = None
            label: int | None = None

            if first["state"] == STATE_NORMAL:
                dataset = "pre-diabetes"
                label = 0 if second["state"] == STATE_NORMAL else 1

            elif first["state"] == STATE_PRE:
                dataset = "diabetes"
                label = 1 if second["state"] == STATE_DM else 0

            else:
                continue

            pair_transition = f"{first['state']} → {second['state']}"
            cur_g = float(first["glucose"])
            fut_g = float(second["glucose"])
            rows.append({
                "user_key": user,
                "current_checkup_date": first[self.date_col].strftime("%Y-%m-%d"),
                "future_checkup_date": second[self.date_col].strftime("%Y-%m-%d"),
                "dataset": dataset,
                "label": label,
                "selected_transition": pair_transition,
                "full_transition": pair_transition,
                "current_glucose": cur_g,
                "future_glucose": fut_g,
                "glucose_change": round(fut_g - cur_g, 1),
            })

        df = pd.DataFrame(rows)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records = {}
            for _, row in df.iterrows():
                entry: dict = {
                    "user_key": row["user_key"],
                    "current_checkup_date": row["current_checkup_date"],
                    "future_checkup_date": row["future_checkup_date"],
                    "dataset": row["dataset"],
                    "label": int(row["label"]),
                    "selected_transition": row["selected_transition"],
                    "full_transition": row["full_transition"],
                    "current_glucose": row["current_glucose"],
                    "future_glucose": row["future_glucose"],
                    "glucose_change": row["glucose_change"],
                }
                if drug_lookup is not None:
                    entry["diabetes_drugs"] = drug_lookup.get(row["user_key"], [])
                records[row["user_key"]] = entry
            import json
            output_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Saved: {output_path}  ({len(df):,}명)")

        return df

    def export_multi_visit_dataset(
        self,
        min_visits: int = 3,
        output_path: Path | None = None,
        drug_lookup: dict[str, list[dict[str, str]]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """3회 이상 수검자를 pre-diabetes / diabetes 데이터셋으로 분류하여 반환하고 JSON으로 저장한다.

        인접한 (i, i+1) 쌍 외에도, 중간 스텝을 포함하는 (i, j) 다중 스텝 쌍(label=0만)을 추가한다.

        분류 기준:
            pre-diabetes
                - 유효 조건: state[i] == 정상, state[i-1..] ∉ {전단계, 당뇨}
                - label=0 (단일): state[i+1] == 정상
                - label=1 (단일): state[i+1] ∈ {전단계, 당뇨}
                - label=0 (다중): state[j] == 정상, state[i+1..j-1] 모두 정상 (j > i+1)
            diabetes
                - 유효 조건: state[i] == 전단계, state[i-1..] ≠ 당뇨
                - label=0 (단일): state[i+1] ∈ {전단계, 정상}
                - label=1 (단일): state[i+1] == 당뇨
                - label=0 (다중): state[j] ∈ {전단계, 정상}, state[i+1..j-1] 모두 당뇨 아님 (j > i+1)

        JSON 키는 ``{dataset}::{user_key}::{current_checkup_date}::{future_checkup_date}`` 형태로 고유성을 보장한다.

        Args:
            min_visits: 최소 방문 횟수 (기본 3).
            output_path: 저장할 JSON 파일 경로. None이면 저장하지 않는다.
            drug_lookup: MedicalHistoryLoader.load_drug_lookup()의 반환값.
                None이면 약물 정보를 포함하지 않는다.

        Returns:
            {"pre-diabetes": DataFrame, "diabetes": DataFrame}
        """
        visit_counts = self._labeled.groupby(self.user_col).size()
        target_users = visit_counts[visit_counts >= min_visits].index
        subset = self._labeled[self._labeled[self.user_col].isin(target_users)]

        full_transitions: dict[str, str] = {
            user: " → ".join(grp.sort_values(self.date_col)["state"].tolist())
            for user, grp in subset.groupby(self.user_col)
        }

        candidates: dict[str, list[dict]] = {"pre-diabetes": [], "diabetes": []}

        for user, group in subset.groupby(self.user_col):
            grp = (
                group.sort_values(self.date_col)
                .dropna(subset=[self.date_col, "glucose"])
                .reset_index(drop=True)
            )
            states = grp["state"].tolist()
            dates = grp[self.date_col].tolist()
            glucoses = grp["glucose"].tolist()

            for i in range(len(states) - 1):
                prev_states = states[:i]  # 현재 시점 이전의 모든 상태
                curr, nxt = states[i], states[i + 1]

                base = {
                    "user_key": user,
                    "current_checkup_date": dates[i].strftime("%Y-%m-%d"),
                    "future_checkup_date": dates[i + 1].strftime("%Y-%m-%d"),
                    "current_glucose": glucoses[i],
                    "future_glucose": glucoses[i + 1],
                    "full_transition": full_transitions[user],
                    "selected_transition": f"{curr} → {nxt}",
                }

                # pre-diabetes: 과거 전체에 전단계/당뇨 이력 없어야 함
                no_prior_dm = not any(s in (STATE_PRE, STATE_DM) for s in prev_states)
                if curr == STATE_NORMAL and no_prior_dm:
                    if nxt == STATE_NORMAL:
                        candidates["pre-diabetes"].append({**base, "dataset": "pre-diabetes", "label": 0})
                    elif nxt in (STATE_PRE, STATE_DM):
                        candidates["pre-diabetes"].append({**base, "dataset": "pre-diabetes", "label": 1})

                # diabetes: 과거 전체에 당뇨 이력 없어야 함
                no_prior_full_dm = not any(s == STATE_DM for s in prev_states)
                if curr == STATE_PRE and no_prior_full_dm:
                    if nxt in (STATE_PRE, STATE_NORMAL):
                        candidates["diabetes"].append({**base, "dataset": "diabetes", "label": 0})
                    elif nxt == STATE_DM:
                        candidates["diabetes"].append({**base, "dataset": "diabetes", "label": 1})

            # 다중 스텝 전이 (label=0만): j > i+1인 (i, j) 쌍
            for i in range(len(states) - 2):
                prev_states = states[:i]
                curr = states[i]
                no_prior_dm = not any(s in (STATE_PRE, STATE_DM) for s in prev_states)
                no_prior_full_dm = not any(s == STATE_DM for s in prev_states)

                for j in range(i + 2, len(states)):
                    intermediate = states[i + 1:j]
                    nxt = states[j]
                    sel_trans = " → ".join(states[i:j + 1])

                    multi_base = {
                        "user_key": user,
                        "current_checkup_date": dates[i].strftime("%Y-%m-%d"),
                        "future_checkup_date": dates[j].strftime("%Y-%m-%d"),
                        "current_glucose": glucoses[i],
                        "future_glucose": glucoses[j],
                        "full_transition": full_transitions[user],
                        "selected_transition": sel_trans,
                    }

                    # pre-diabetes label=0: 중간에 전단계/당뇨 없고 최종이 정상
                    if (curr == STATE_NORMAL and no_prior_dm and
                            nxt == STATE_NORMAL and
                            not any(s in (STATE_PRE, STATE_DM) for s in intermediate)):
                        candidates["pre-diabetes"].append({**multi_base, "dataset": "pre-diabetes", "label": 0})

                    # diabetes label=0: 중간에 당뇨 없고 최종이 전단계/정상
                    if (curr == STATE_PRE and no_prior_full_dm and
                            nxt in (STATE_PRE, STATE_NORMAL) and
                            not any(s == STATE_DM for s in intermediate)):
                        candidates["diabetes"].append({**multi_base, "dataset": "diabetes", "label": 0})

        result: dict[str, pd.DataFrame] = {}
        for ds, cands in candidates.items():
            if not cands:
                result[ds] = pd.DataFrame()
                continue
            df_c = pd.DataFrame(cands)
            df_c["glucose_change"] = (
                df_c["future_glucose"] - df_c["current_glucose"]
            ).round(1)
            result[ds] = (
                df_c.sort_values([self.user_col, "current_checkup_date"])
                .reset_index(drop=True)
            )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records: dict[str, dict] = {}
            for ds, df_out in result.items():
                for _, row in df_out.iterrows():
                    key = f"{ds}::{row['user_key']}::{row['current_checkup_date']}::{row['future_checkup_date']}"
                    entry: dict = {
                        "user_key": row["user_key"],
                        "current_checkup_date": row["current_checkup_date"],
                        "future_checkup_date": row["future_checkup_date"],
                        "dataset": row["dataset"],
                        "label": int(row["label"]),
                        "selected_transition": row["selected_transition"],
                        "full_transition": row["full_transition"],
                        "current_glucose": row["current_glucose"],
                        "future_glucose": row["future_glucose"],
                        "glucose_change": row["glucose_change"],
                    }
                    if drug_lookup is not None:
                        entry["diabetes_drugs"] = drug_lookup.get(row["user_key"], [])
                    records[key] = entry
            import json
            output_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for ds, df_out in result.items():
                l0 = int((df_out["label"] == 0).sum())
                l1 = int((df_out["label"] == 1).sum())
                print(f"[{ds}] 총 {len(df_out):,}건  (label=0: {l0:,}, label=1: {l1:,})")
            print(f"Saved: {output_path}")

        return result

    def plot_transition(self, output_dir: Path | None = None, show: bool = True) -> None:
        """전이 행렬 히트맵, 전이 유형 막대 차트, 2회 수검자 시퀀스 패턴을 시각화한다.

        Args:
            output_dir: 저장 경로. None이면 파일로 저장하지 않는다.
            show: True이면 plt.show()를 호출한다.
        """
        summary = self.summary()
        matrix = self._matrix
        seq2 = self.sequence_summary(n_visits=[2]).get(2, pd.DataFrame())
        n2_users = int((self._labeled.groupby(self.user_col).size() == 2).sum())

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("공복혈당 기반 당뇨 상태 전이 분석", fontsize=14, fontweight="bold")

        # ── ① 전이 확률 행렬 히트맵 ──────────────────────────
        ax = axes[0]
        data = matrix.values.astype(float)
        row_sums = data.sum(axis=1, keepdims=True)
        pct_matrix = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0) * 100

        im = ax.imshow(pct_matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(STATES)))
        ax.set_yticks(range(len(STATES)))
        ax.set_xticklabels(STATES, fontsize=10)
        ax.set_yticklabels(STATES, fontsize=10)
        ax.set_xlabel("다음 상태 (To)")
        ax.set_ylabel("현재 상태 (From)")
        ax.set_title("전이 확률 행렬 (행 기준 %)")
        for i in range(len(STATES)):
            for j in range(len(STATES)):
                count = int(data[i, j])
                pct = pct_matrix[i, j]
                color = "white" if pct > 55 else "black"
                ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, color=color)
        plt.colorbar(im, ax=ax, label="%")

        # ── ② 전이 유형별 건수 (전체) ─────────────────────────
        ax2 = axes[1]
        counts_val = summary["count"].tolist()
        bar_colors = [STATE_COLORS.get(r["from_state"], "#888888") for _, r in summary.iterrows()]
        bars = ax2.barh(summary["label"].tolist()[::-1], counts_val[::-1],
                        color=bar_colors[::-1], edgecolor="white")
        ax2.set_title("전이 유형별 건수 (전체)")
        ax2.set_xlabel("건수")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.set_xlim(right=max(counts_val) * 1.3)
        for bar, count, pct in zip(bars, counts_val[::-1], summary["pct"].tolist()[::-1]):
            ax2.text(bar.get_width() + max(counts_val) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{count:,}건 ({pct}%)", va="center", fontsize=9)
        legend_patches = [mpatches.Patch(color=STATE_COLORS[s], label=s) for s in STATES]
        ax2.legend(handles=legend_patches, loc="lower right", fontsize=9)

        # ── ③ 2회 수검자 시퀀스 패턴 ─────────────────────────
        ax3 = axes[2]
        if not seq2.empty:
            bar_colors3 = [
                STATE_COLORS.get(row["sequence"].split(" → ")[0], "#888888")
                for _, row in seq2.iterrows()
            ]
            bars3 = ax3.barh(seq2["sequence"].tolist()[::-1], seq2["count"].tolist()[::-1],
                             color=bar_colors3[::-1], edgecolor="white")
            ax3.set_title(f"2회 수검자 시퀀스 ({n2_users:,}명)")
            ax3.set_xlabel("명수")
            ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax3.set_xlim(right=seq2["count"].max() * 1.35)
            for bar, (_, row) in zip(bars3, seq2.iloc[::-1].iterrows()):
                ax3.text(bar.get_width() + seq2["count"].max() * 0.03,
                         bar.get_y() + bar.get_height() / 2,
                         f"{row['count']:,}명 ({row['pct']}%)", va="center", fontsize=9)

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "glucose_transition.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


class CombinedGlucoseTransitionAnalyzer(GlucoseTransitionAnalyzer):
    """종합·국가 검진을 결합하여 공복혈당 기준 상태 전이를 분석한다.

    공복혈당 기록이 있는 종합·국가 검진 이력을 모두 합산하여 분석 대상으로 사용한다.
    시각화(summary, plot_transition, plot_sequences)는 출처를 제거한 순수 상태명을 사용한다.
    JSON export(export_dataset, export_multi_visit_dataset)의 selected_transition /
    full_transition 필드에는 ``정상(종합) → 정상(국가)`` 형식으로 출처를 표기하며,
    current_checkup은 반드시 종합검진 이력이어야 한다.
    """

    SOURCE_TOTAL = "종합"
    SOURCE_NATIONAL = "국가"

    def __init__(
        self,
        total_df: pd.DataFrame,
        national_df: pd.DataFrame,
        date_col: str = "checkup_date",
        user_col: str = "user_key",
        detail_col: str = "detail_infos",
    ) -> None:
        """
        Args:
            total_df: 종합검진 DataFrame (DataLoader.load() 결과)
            national_df: 국가검진 DataFrame (DataLoader.load_streaming() 결과)
            date_col: 검진일 컬럼명
            user_col: 사용자 식별 컬럼명
            detail_col: 세부 검진 항목 컬럼명
        """
        self.date_col = date_col
        self.user_col = user_col
        self.min_visits = 1

        total_labeled = self._extract_glucose_labeled(
            total_df, user_col, date_col, detail_col, self.SOURCE_TOTAL
        )
        national_labeled = self._extract_glucose_labeled(
            national_df, user_col, date_col, detail_col, self.SOURCE_NATIONAL
        )
        combined = pd.concat([total_labeled, national_labeled], ignore_index=True)

        self._labeled = (
            combined
            .sort_values([user_col, date_col])
            .reset_index(drop=True)
        )
        self._transitions = self._build_transitions()
        self._matrix = self._build_matrix()

    @staticmethod
    def _filter_eligible_users(
        total_labeled: pd.DataFrame,
        combined: pd.DataFrame,
        user_col: str,
        date_col: str,
    ) -> pd.Index:
        """종합검진 이후 추가 수검 이력(공복혈당 있음)이 존재하는 수검자를 반환한다."""
        first_total = total_labeled.groupby(user_col)[date_col].min().rename("first_total")
        last_any = combined.groupby(user_col)[date_col].max().rename("last_any")
        joined = first_total.to_frame().join(last_any, how="left")
        return joined[joined["last_any"] > joined["first_total"]].index

    def _build_transitions(self) -> pd.DataFrame:
        """연속 검진 쌍의 상태 전이를 생성한다 (from_source / to_source 컬럼 포함)."""
        rows = []
        for user, group in self._labeled.groupby(self.user_col):
            grp = group.sort_values(self.date_col)
            states = grp["state"].tolist()
            sources = grp["source"].tolist()
            dates = grp[self.date_col].tolist()
            for i in range(len(states) - 1):
                rows.append({
                    self.user_col: user,
                    "from_state": states[i],
                    "from_source": sources[i],
                    "to_state": states[i + 1],
                    "to_source": sources[i + 1],
                    "from_date": dates[i],
                    "to_date": dates[i + 1],
                })
        return pd.DataFrame(rows)

    def export_dataset(
        self,
        output_path: Path | None = None,
        drug_lookup: dict[str, list[dict[str, str]]] | None = None,
    ) -> pd.DataFrame:
        """2회 수검자 중 첫 상태가 '정상' 또는 '당뇨병전단계'인 수검자를 분류하여 JSON으로 저장한다.

        selected_transition / full_transition 에 이력 출처(종합/국가)를 괄호로 표기한다.
        예) 정상(종합) → 정상(국가)

        분류 기준:
            - "pre-diabetes" : 첫 상태 == 정상
                label=0 : 정상 → 정상
                label=1 : 정상 → 당뇨병전단계  또는  정상 → 당뇨병
            - "diabetes"     : 첫 상태 == 당뇨병전단계
                label=0 : 당뇨병전단계 → 당뇨병전단계  또는  당뇨병전단계 → 정상
                label=1 : 당뇨병전단계 → 당뇨병
        """
        visit_counts = self._labeled.groupby(self.user_col).size()
        two_visit_users = visit_counts[visit_counts == 2].index
        subset = self._labeled[self._labeled[self.user_col].isin(two_visit_users)]

        rows = []
        for user, group in subset.groupby(self.user_col):
            group = group.sort_values(self.date_col).dropna(subset=[self.date_col])
            if len(group) != 2:
                continue
            first, second = group.iloc[0], group.iloc[1]
            if pd.isna(first[self.date_col]) or pd.isna(second[self.date_col]):
                continue

            dataset: str | None = None
            label: int | None = None

            if first["state"] == STATE_NORMAL:
                dataset = "pre-diabetes"
                label = 0 if second["state"] == STATE_NORMAL else 1
            elif first["state"] == STATE_PRE:
                dataset = "diabetes"
                label = 1 if second["state"] == STATE_DM else 0
            else:
                continue

            pair_transition = (
                f"{first['state']}({first['source']}) → "
                f"{second['state']}({second['source']})"
            )
            cur_g = float(first["glucose"])
            fut_g = float(second["glucose"])
            rows.append({
                "user_key": user,
                "current_checkup_date": first[self.date_col].strftime("%Y-%m-%d"),
                "future_checkup_date": second[self.date_col].strftime("%Y-%m-%d"),
                "dataset": dataset,
                "label": label,
                "selected_transition": pair_transition,
                "full_transition": pair_transition,
                "current_glucose": cur_g,
                "future_glucose": fut_g,
                "glucose_change": round(fut_g - cur_g, 1),
            })

        df = pd.DataFrame(rows)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records = {}
            for _, row in df.iterrows():
                entry: dict = {
                    "user_key": row["user_key"],
                    "current_checkup_date": row["current_checkup_date"],
                    "future_checkup_date": row["future_checkup_date"],
                    "dataset": row["dataset"],
                    "label": int(row["label"]),
                    "selected_transition": row["selected_transition"],
                    "full_transition": row["full_transition"],
                    "current_glucose": row["current_glucose"],
                    "future_glucose": row["future_glucose"],
                    "glucose_change": row["glucose_change"],
                }
                if drug_lookup is not None:
                    entry["diabetes_drugs"] = drug_lookup.get(row["user_key"], [])
                records[row["user_key"]] = entry
            import json
            output_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Saved: {output_path}  ({len(df):,}명)")

        return df

    def export_multi_visit_dataset(
        self,
        min_visits: int = 3,
        output_path: Path | None = None,
        drug_lookup: dict[str, list[dict[str, str]]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """3회 이상 수검자를 pre-diabetes / diabetes 데이터셋으로 분류하여 JSON으로 저장한다.

        selected_transition / full_transition 에 이력 출처(종합/국가)를 괄호로 표기한다.
        예) 정상(종합) → 정상(국가) → 정상(종합)

        인접한 (i, i+1) 쌍 외에도, 중간 스텝을 포함하는 (i, j) 다중 스텝 쌍(label=0만)을 추가한다.
        분류 기준은 GlucoseTransitionAnalyzer.export_multi_visit_dataset 과 동일하다.
        """
        visit_counts = self._labeled.groupby(self.user_col).size()
        target_users = visit_counts[visit_counts >= min_visits].index
        subset = self._labeled[self._labeled[self.user_col].isin(target_users)]

        full_transitions: dict[str, str] = {
            user: " → ".join(
                f"{s}({src})"
                for s, src in zip(
                    grp.sort_values(self.date_col)["state"],
                    grp.sort_values(self.date_col)["source"],
                )
            )
            for user, grp in subset.groupby(self.user_col)
        }

        candidates: dict[str, list[dict]] = {"pre-diabetes": [], "diabetes": []}

        for user, group in subset.groupby(self.user_col):
            grp = (
                group.sort_values(self.date_col)
                .dropna(subset=[self.date_col, "glucose"])
                .reset_index(drop=True)
            )
            states = grp["state"].tolist()
            sources = grp["source"].tolist()
            dates = grp[self.date_col].tolist()
            glucoses = grp["glucose"].tolist()

            for i in range(len(states) - 1):
                prev_states = states[:i]
                curr, nxt = states[i], states[i + 1]
                curr_src, nxt_src = sources[i], sources[i + 1]

                base = {
                    "user_key": user,
                    "current_checkup_date": dates[i].strftime("%Y-%m-%d"),
                    "future_checkup_date": dates[i + 1].strftime("%Y-%m-%d"),
                    "current_glucose": glucoses[i],
                    "future_glucose": glucoses[i + 1],
                    "full_transition": full_transitions[user],
                    "selected_transition": f"{curr}({curr_src}) → {nxt}({nxt_src})",
                }

                no_prior_dm = not any(s in (STATE_PRE, STATE_DM) for s in prev_states)
                if curr == STATE_NORMAL and no_prior_dm:
                    if nxt == STATE_NORMAL:
                        candidates["pre-diabetes"].append({**base, "dataset": "pre-diabetes", "label": 0})
                    elif nxt in (STATE_PRE, STATE_DM):
                        candidates["pre-diabetes"].append({**base, "dataset": "pre-diabetes", "label": 1})

                no_prior_full_dm = not any(s == STATE_DM for s in prev_states)
                if curr == STATE_PRE and no_prior_full_dm:
                    if nxt in (STATE_PRE, STATE_NORMAL):
                        candidates["diabetes"].append({**base, "dataset": "diabetes", "label": 0})
                    elif nxt == STATE_DM:
                        candidates["diabetes"].append({**base, "dataset": "diabetes", "label": 1})

            # 다중 스텝 전이 (label=0만): j > i+1인 (i, j) 쌍
            for i in range(len(states) - 2):
                prev_states = states[:i]
                curr = states[i]
                no_prior_dm = not any(s in (STATE_PRE, STATE_DM) for s in prev_states)
                no_prior_full_dm = not any(s == STATE_DM for s in prev_states)

                for j in range(i + 2, len(states)):
                    intermediate = states[i + 1:j]
                    nxt = states[j]
                    sel_trans = " → ".join(
                        f"{states[k]}({sources[k]})" for k in range(i, j + 1)
                    )

                    multi_base = {
                        "user_key": user,
                        "current_checkup_date": dates[i].strftime("%Y-%m-%d"),
                        "future_checkup_date": dates[j].strftime("%Y-%m-%d"),
                        "current_glucose": glucoses[i],
                        "future_glucose": glucoses[j],
                        "full_transition": full_transitions[user],
                        "selected_transition": sel_trans,
                    }

                    # pre-diabetes label=0: 중간에 전단계/당뇨 없고 최종이 정상
                    if (curr == STATE_NORMAL and no_prior_dm and
                            nxt == STATE_NORMAL and
                            not any(s in (STATE_PRE, STATE_DM) for s in intermediate)):
                        candidates["pre-diabetes"].append({**multi_base, "dataset": "pre-diabetes", "label": 0})

                    # diabetes label=0: 중간에 당뇨 없고 최종이 전단계/정상
                    if (curr == STATE_PRE and no_prior_full_dm and
                            nxt in (STATE_PRE, STATE_NORMAL) and
                            not any(s == STATE_DM for s in intermediate)):
                        candidates["diabetes"].append({**multi_base, "dataset": "diabetes", "label": 0})

        result: dict[str, pd.DataFrame] = {}
        for ds, cands in candidates.items():
            if not cands:
                result[ds] = pd.DataFrame()
                continue
            df_c = pd.DataFrame(cands)
            df_c["glucose_change"] = (df_c["future_glucose"] - df_c["current_glucose"]).round(1)
            result[ds] = (
                df_c.sort_values([self.user_col, "current_checkup_date"])
                .reset_index(drop=True)
            )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            records: dict[str, dict] = {}
            for ds, df_out in result.items():
                for _, row in df_out.iterrows():
                    key = f"{ds}::{row['user_key']}::{row['current_checkup_date']}::{row['future_checkup_date']}"
                    entry: dict = {
                        "user_key": row["user_key"],
                        "current_checkup_date": row["current_checkup_date"],
                        "future_checkup_date": row["future_checkup_date"],
                        "dataset": row["dataset"],
                        "label": int(row["label"]),
                        "selected_transition": row["selected_transition"],
                        "full_transition": row["full_transition"],
                        "current_glucose": row["current_glucose"],
                        "future_glucose": row["future_glucose"],
                        "glucose_change": row["glucose_change"],
                    }
                    if drug_lookup is not None:
                        entry["diabetes_drugs"] = drug_lookup.get(row["user_key"], [])
                    records[key] = entry
            import json
            output_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for ds, df_out in result.items():
                l0 = int((df_out["label"] == 0).sum())
                l1 = int((df_out["label"] == 1).sum())
                print(f"[{ds}] 총 {len(df_out):,}건  (label=0: {l0:,}, label=1: {l1:,})")
            print(f"Saved: {output_path}")

        return result
