"""하이퍼파라미터 랜덤 탐색 실행기."""
from __future__ import annotations

import contextlib
import io
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.dl_trainer import CVResult, DLConfig, NestedCVTrainer, ResultExporter


class HParamSearcher:
    """랜덤 하이퍼파라미터 탐색 (Random Search) 실행기.

    Args:
        output_root: 탐색 결과 루트 디렉토리.
        n_folds: Nested CV outer fold 수.
        max_epochs: 최대 에폭 수.
        early_stopping: 조기 종료 활성화 여부.
        patience: 조기 종료 전 개선 없는 최대 에폭 수.
        loss: 손실 함수 종류 ('bce' | 'weighted_bce').
    """

    def __init__(
        self,
        output_root: Path,
        n_folds: int = 4,
        max_epochs: int = 1000,
        early_stopping: bool = True,
        patience: int = 40,
        loss: str = "weighted_bce",
    ) -> None:
        self.output_root = output_root
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.loss = loss

    @staticmethod
    @contextlib.contextmanager
    def _silent():  # type: ignore[return]
        """stdout / stderr / warnings 를 모두 억제하는 컨텍스트 매니저."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

    def load_best_f1(self, output_dir: Path) -> float:
        """저장된 best_config.json에서 mean_f1을 로드한다.

        Args:
            output_dir: best_config.json이 위치한 디렉토리.

        Returns:
            저장된 mean_f1 값. 파일이 없으면 -1.0.
        """
        path = output_dir / "best_config.json"
        if not path.exists():
            return -1.0
        return float(json.loads(path.read_text(encoding="utf-8")).get("mean_f1", -1.0))

    def run_trial(
        self,
        params: dict[str, Any],
        dataset_path: Path,
        vif_path: Path,
        output_dir: Path,
        features: list[str],
    ) -> tuple[float, CVResult, DLConfig]:
        """단일 하이퍼파라미터 조합으로 Nested CV를 실행한다.

        Args:
            params: 탐색 공간에서 샘플링된 하이퍼파라미터 딕셔너리.
            dataset_path: 데이터셋 Excel 경로.
            vif_path: VIF 생존 피처 CSV 경로.
            output_dir: 결과 저장 디렉토리.
            features: 사용할 피처 목록.

        Returns:
            (mean_f1, CVResult, DLConfig) 튜플.
        """
        config = DLConfig(
            dataset_path=dataset_path,
            vif_features_path=vif_path,
            output_dir=output_dir,
            label_col="label",
            n_outer_folds=self.n_folds,
            random_state=params["random_state"],
            imputation_strategy=params["imputation_strategy"],
            knn_n_neighbors=params["knn_n_neighbors"],
            mice_max_iter=params["mice_max_iter"],
            imputation_fill_value=params["imputation_fill_value"],
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            dropout_rate=params["dropout_rate"],
            loss=self.loss,
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=self.max_epochs,
            early_stopping=self.early_stopping,
            patience=self.patience,
            validation_fraction=params["validation_fraction"],
            selected_features=features,
        )
        trainer = NestedCVTrainer(config)
        with self._silent():
            trainer.prepare_data()
            result = trainer.run()
        mean_f1 = float(result.summary().set_index("metric").loc["f1", "mean"])
        return mean_f1, result, config

    def save_best(
        self,
        result: CVResult,
        config: DLConfig,
        mean_f1: float,
        params: dict[str, Any],
        trial: int,
    ) -> None:
        """최고 성능 결과를 output_dir에 저장한다.

        모델 아티팩트, 평가 지표, 시각화, best_config.json을 모두 저장한다.

        Args:
            result: Nested CV 결과.
            config: 해당 trial의 DLConfig.
            mean_f1: fold 평균 F1 점수.
            params: 해당 trial의 하이퍼파라미터.
            trial: trial 번호.
        """
        exporter = ResultExporter(config.output_dir)
        with self._silent():
            exporter.save_artifacts(result, config)
            exporter.save_metrics_csv(result)
            exporter.save_summary_csv(result)
            exporter.save_confusion_matrices(result)
            exporter.save_roc_curve(result)
            exporter.save_pr_curve(result)
            exporter.save_training_history_csv(result)
            exporter.save_learning_curves(result)

        best_info = {
            "trial": trial,
            "mean_f1": mean_f1,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            "fold_metrics": result.metrics_df().to_dict(orient="records"),
            "timestamp": datetime.now().isoformat(),
        }
        (config.output_dir / "best_config.json").write_text(
            json.dumps(best_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def search(
        self,
        n_trials: int,
        param_grid: dict[str, list[Any]],
        log_path: Path,
        diabetes_data: Path,
        diabetes_vif: Path,
        diabetes_features: list[str],
        prediabetes_data: Path,
        prediabetes_vif: Path,
        prediabetes_features: list[str],
    ) -> pd.DataFrame:
        """랜덤 탐색을 실행한다. 이전 탐색 기록이 있으면 이어서 진행한다.

        Args:
            n_trials: 이번에 실행할 trial 횟수.
            param_grid: 탐색할 하이퍼파라미터 공간.
            log_path: 탐색 로그 CSV 저장 경로.
            diabetes_data: 당뇨병 데이터셋 경로.
            diabetes_vif: 당뇨병 VIF 피처 CSV 경로.
            diabetes_features: 당뇨병 사용 피처 목록.
            prediabetes_data: 당뇨병전단계 데이터셋 경로.
            prediabetes_vif: 당뇨병전단계 VIF 피처 CSV 경로.
            prediabetes_features: 당뇨병전단계 사용 피처 목록.

        Returns:
            전체 탐색 로그 DataFrame.
        """
        from IPython.display import clear_output, display

        existing_log = pd.read_csv(log_path).to_dict(orient="records") if log_path.exists() else []
        trial_offset = len(existing_log)

        best_f1 = {
            "diabetes":     self.load_best_f1(self.output_root / "diabetes"),
            "pre_diabetes": self.load_best_f1(self.output_root / "pre_diabetes"),
        }

        print(f"이어서 탐색: {trial_offset}번부터 시작")
        print(f"현재 최고 F1  |  Diabetes: {best_f1['diabetes']:.4f}  |  Pre-diabetes: {best_f1['pre_diabetes']:.4f}")

        trial_logs: list[dict] = []

        for i in range(1, n_trials + 1):
            trial  = trial_offset + i
            params = {k: random.choice(v) for k, v in param_grid.items()}
            start  = datetime.now()
            notes: list[str] = []

            row: dict = {
                "trial": trial,
                **{f"p_{k}": (str(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            }

            try:
                d_f1, d_result, d_config = self.run_trial(
                    params, diabetes_data, diabetes_vif,
                    self.output_root / "diabetes", diabetes_features,
                )
                row["diabetes_f1"]   = round(d_f1, 6)
                row["diabetes_best"] = d_f1 > best_f1["diabetes"]
                if d_f1 > best_f1["diabetes"]:
                    best_f1["diabetes"] = d_f1
                    self.save_best(d_result, d_config, d_f1, params, trial)
                    notes.append(f"Diabetes  new best  F1={d_f1:.4f}  [saved]")
            except Exception as e:
                row["diabetes_f1"], row["diabetes_best"] = float("nan"), False
                notes.append(f"Diabetes  ERROR: {e}")

            try:
                p_f1, p_result, p_config = self.run_trial(
                    params, prediabetes_data, prediabetes_vif,
                    self.output_root / "pre_diabetes", prediabetes_features,
                )
                row["prediabetes_f1"]   = round(p_f1, 6)
                row["prediabetes_best"] = p_f1 > best_f1["pre_diabetes"]
                if p_f1 > best_f1["pre_diabetes"]:
                    best_f1["pre_diabetes"] = p_f1
                    self.save_best(p_result, p_config, p_f1, params, trial)
                    notes.append(f"Pre-diabetes  new best  F1={p_f1:.4f}  [saved]")
            except Exception as e:
                row["prediabetes_f1"], row["prediabetes_best"] = float("nan"), False
                notes.append(f"Pre-diabetes  ERROR: {e}")

            row["elapsed_sec"] = round((datetime.now() - start).total_seconds(), 1)
            row["timestamp"]   = datetime.now().isoformat()
            trial_logs.append(row)

            all_logs = existing_log + trial_logs
            pd.DataFrame(all_logs).to_csv(log_path, index=False)

            clear_output(wait=True)
            total = trial_offset + n_trials
            print(f"Trial {trial:>4d} / {total}   ({row['elapsed_sec']:.1f}s)")
            print(f"  Diabetes     F1 = {row.get('diabetes_f1', float('nan')):.4f}   best = {best_f1['diabetes']:.4f}")
            print(f"  Pre-diabetes F1 = {row.get('prediabetes_f1', float('nan')):.4f}   best = {best_f1['pre_diabetes']:.4f}")
            for note in notes:
                print(f"  ★ {note}")
            print(
                f"  params: impute={params['imputation_strategy']}  layers={params['hidden_layer_sizes']}  "
                f"lr={params['learning_rate']}  dr={params['dropout_rate']}  bs={params['batch_size']}"
            )
            print()

            log_df  = pd.DataFrame(all_logs)
            cols    = [
                "trial", "p_imputation_strategy", "p_hidden_layer_sizes",
                "p_learning_rate", "p_dropout_rate", "p_batch_size",
                "diabetes_f1", "prediabetes_f1", "diabetes_best", "prediabetes_best",
            ]
            recent  = log_df[[c for c in cols if c in log_df.columns]].tail(20).copy()
            recent.columns = [
                c.replace("p_imputation_", "imp_").replace("p_hidden_layer_", "layers").replace("p_", "")
                for c in recent.columns
            ]
            f1c = [c for c in recent.columns if c.endswith("_f1")]
            display(
                recent.style
                .format({c: "{:.4f}" for c in f1c}, na_rep="—")
                .highlight_max(subset=f1c, color="lightgreen")
            )

        print(f"\n탐색 완료!  Diabetes best={best_f1['diabetes']:.4f}  Pre-diabetes best={best_f1['pre_diabetes']:.4f}")
        return pd.DataFrame(all_logs)

    def show_best(self) -> None:
        """저장된 best_config.json의 최적 결과를 출력한다."""
        from IPython.display import display

        for tag, subdir in [("Diabetes", "diabetes"), ("Pre-diabetes", "pre_diabetes")]:
            path = self.output_root / subdir / "best_config.json"
            if not path.exists():
                print(f"{tag}: 저장된 결과 없음")
                continue
            best  = json.loads(path.read_text(encoding="utf-8"))
            mf1   = best["mean_f1"]
            trial = best["trial"]
            ts    = best["timestamp"][:19]
            print(f"{'='*60}")
            print(f"{tag}  |  Mean F1 = {mf1:.4f}  |  Trial #{trial}  |  {ts}")
            print(f"{'='*60}")
            for k, v in best["params"].items():
                print(f"  {k:<25} = {v}")
            print()
            display(pd.DataFrame(best["fold_metrics"]).set_index("fold"))
            print()

    def show_top10(self, log_path: Path) -> None:
        """탐색 로그에서 F1 기준 상위 10개 trial을 출력한다.

        Args:
            log_path: 탐색 로그 CSV 경로.
        """
        from IPython.display import display

        if not log_path.exists():
            print("탐색 로그 없음")
            return

        log_df = pd.read_csv(log_path)
        print(f"총 {len(log_df)}회 시도")

        base_cols = [
            "trial", "p_imputation_strategy", "p_hidden_layer_sizes",
            "p_learning_rate", "p_dropout_rate", "p_batch_size",
            "p_validation_fraction", "p_random_state",
            "diabetes_f1", "prediabetes_f1",
        ]
        base_cols = [c for c in base_cols if c in log_df.columns]

        def _rename(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.columns = [
                c.replace("p_imputation_", "imp_").replace("p_hidden_layer_", "layers").replace("p_", "")
                for c in df.columns
            ]
            return df

        for sort_col, label in [("diabetes_f1", "Diabetes"), ("prediabetes_f1", "Pre-diabetes")]:
            if sort_col not in log_df.columns:
                continue
            top = _rename(log_df.nlargest(10, sort_col)[base_cols])
            f1c = [c for c in top.columns if c.endswith("_f1")]
            print(f"── {label} Top-10 ──")
            display(top.style.format({c: "{:.4f}" for c in f1c}).highlight_max(subset=f1c, color="lightgreen"))
            print()
