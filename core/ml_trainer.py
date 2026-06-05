"""Nested 5-Fold Cross-Validation ML Trainer for prediabetes/diabetes transitions."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


ModelName = Literal["svm", "lr", "rf", "xgb", "lgbm", "gb", "et"]
TunerType = Literal["grid", "bayes"]
ImputationStrategy = Literal["knn", "mean", "median", "most_frequent", "constant"]

_META_COLUMNS = frozenset(
    [
        "user_key",
        "current_checkup_date",
        "future_checkup_date",
        "selected_transition",
        "full_transition",
        "interval_days",
        "checkup_date",
        "birthday",
    ]
)

_CATEGORICAL_COLUMNS = [
    "gender",
    "Bilirubin",
    "Blood",
    "Glucose",
    "Keton",
    "Leukocyte",
    "Nitrite",
    "Protein",
    "Urobilinogen",
    "HBs-Ag",
    "HBs-Ab",
]

# 범위 문자열(예: '0-5', '0 - 2')을 중간값으로 변환할 컬럼
_RANGE_COLUMNS = [
    "백혈구(소변현미경)",
    "적혈구(소변현미경)",
]

# BayesSearchCV용 연속/이산 탐색 공간 (GridSearch보다 넓은 범위를 효율적으로 탐색)
_MODEL_BAYES_SPACES: dict[str, dict] = {
    "svm": {
        "classifier__C": Real(1e-3, 1e3, prior="log-uniform"),
        "classifier__kernel": Categorical(["rbf", "linear"]),
        "classifier__gamma": Categorical(["scale", "auto"]),
    },
    "lr": {
        "classifier__C": Real(1e-3, 1e3, prior="log-uniform"),
        "classifier__max_iter": Categorical([1000]),
        "classifier__solver": Categorical(["lbfgs", "saga"]),
    },
    "rf": {
        "classifier__n_estimators": Integer(50, 500),
        "classifier__max_depth": Integer(3, 20),
        "classifier__min_samples_split": Integer(2, 10),
    },
    "xgb": {
        "classifier__n_estimators": Integer(50, 500),
        "classifier__learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
        "classifier__max_depth": Integer(2, 10),
        "classifier__subsample": Real(0.5, 1.0),
        "classifier__colsample_bytree": Real(0.5, 1.0),
    },
    "lgbm": {
        "classifier__n_estimators": Integer(50, 500),
        "classifier__learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
        "classifier__max_depth": Integer(2, 10),
        "classifier__num_leaves": Integer(10, 100),
        "classifier__min_child_samples": Integer(5, 50),
    },
    "gb": {
        "classifier__n_estimators": Integer(50, 500),
        "classifier__learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
        "classifier__max_depth": Integer(2, 8),
        "classifier__subsample": Real(0.5, 1.0),
    },
    "et": {
        "classifier__n_estimators": Integer(50, 500),
        "classifier__max_depth": Integer(3, 20),
        "classifier__min_samples_split": Integer(2, 10),
    },
}

_MODEL_PARAM_GRIDS: dict[str, dict] = {
    "svm": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
    },
    "lr": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__max_iter": [1000],
        "classifier__solver": ["lbfgs", "saga"],
    },
    "rf": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    },
    "xgb": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.7, 1.0],
    },
    "lgbm": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 5, 7],
        "classifier__num_leaves": [15, 31, 63],
    },
    "gb": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 5, 7],
    },
    "et": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    },
}


class FeatureInspector:
    """인코딩 후 사용 가능한 피처 목록을 조회한다."""

    @staticmethod
    def show(dataset_path: Path, title: str = "사용 가능한 피처") -> list[str]:
        """인코딩 후 선택 가능한 컬럼명을 한 줄로 출력하고 목록을 반환한다."""
        df = pd.read_excel(dataset_path)
        drop_cols = [c for c in df.columns if c in _META_COLUMNS]
        df = df.drop(columns=drop_cols, errors="ignore")
        cat_cols = [c for c in _CATEGORICAL_COLUMNS if c in df.columns]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        feature_cols = [c for c in df.columns if c != "label"]
        print(f"── {title} ({len(feature_cols)}개) ──")
        print("[" + ", ".join(f"'{c}'" for c in feature_cols) + "]")
        print()
        return feature_cols


@dataclass
class MLConfig:
    """ML 훈련 파이프라인 설정."""

    dataset_path: Path
    vif_features_path: Path
    output_dir: Path
    label_col: str = "label"
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    knn_n_neighbors: int = 5
    random_state: int = 42
    models: list[ModelName] = field(default_factory=lambda: ["svm", "lr", "rf", "xgb", "lgbm", "gb", "et"])
    scale_features: bool = True
    selected_features: list[str] | None = None
    """직접 지정할 피처 목록. None이면 vif_features_path에서 자동 로드한다."""
    tuner: TunerType = "grid"
    """하이퍼파라미터 탐색 방법: 'grid' (GridSearchCV) 또는 'bayes' (BayesSearchCV)."""
    n_bayes_iter: int = 50
    """BayesSearchCV 탐색 횟수. tuner='bayes'일 때만 적용된다."""
    imputation_strategy: ImputationStrategy = "knn"
    """결측치 처리 전략.
    - 'knn'          : KNNImputer (knn_n_neighbors 값 사용)
    - 'mean'         : 각 피처의 평균으로 대체
    - 'median'       : 각 피처의 중앙값으로 대체
    - 'most_frequent': 각 피처의 최빈값으로 대체
    - 'constant'     : imputation_fill_value 값으로 일괄 대체
    """
    imputation_fill_value: float = 0.0
    """imputation_strategy='constant'일 때 사용할 대체 값."""


@dataclass
class FoldMetrics:
    """단일 fold 평가 지표."""

    fold: int
    model: ModelName
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    best_params: dict


@dataclass
class ModelResult:
    """모델 전체 Nested CV 결과."""

    model: ModelName
    fold_metrics: list[FoldMetrics]
    fpr_list: list[np.ndarray]
    tpr_list: list[np.ndarray]
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    precision_list: list[np.ndarray]
    recall_list: list[np.ndarray]

    def metrics_df(self) -> pd.DataFrame:
        """fold별 지표 DataFrame."""
        return pd.DataFrame(
            [
                {
                    "fold": m.fold,
                    "model": m.model,
                    "accuracy": m.accuracy,
                    "sensitivity": m.sensitivity,
                    "specificity": m.specificity,
                    "precision": m.precision,
                    "f1": m.f1,
                    "auc": m.auc,
                    "pr_auc": m.pr_auc,
                }
                for m in self.fold_metrics
            ]
        )

    def summary(self) -> pd.DataFrame:
        """지표 평균 ± 표준편차 요약."""
        df = self.metrics_df().drop(columns=["fold", "model"])
        return pd.DataFrame(
            {
                "metric": df.columns,
                "mean": df.mean().values,
                "std": df.std().values,
                "mean±std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(df.mean(), df.std())],
            }
        )


class DataPreparator:
    """데이터 로딩 및 인코딩 전처리."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self._feature_cols: list[str] = []

    def load_and_prepare(self) -> tuple[pd.DataFrame, pd.Series]:
        """데이터를 로딩하고 인코딩 후 (X, y)를 반환한다."""
        df = pd.read_excel(self.config.dataset_path)

        if self.config.selected_features is not None:
            features = self.config.selected_features
            print(f"피처 소스: 직접 지정 ({len(features)}개)")
        else:
            features = self._load_vif_features()
            print(f"피처 소스: VIF CSV ({len(features)}개) — {self.config.vif_features_path.name}")

        df = self._drop_meta(df)
        df = self._encode_categoricals(df)
        df = self._coerce_remaining_objects(df)

        y = df[self.config.label_col].astype(int)
        X = self._select_features(df, features)

        self._feature_cols = X.columns.tolist()
        return X, y

    @property
    def feature_cols(self) -> list[str]:
        """선택된 피처 컬럼명 목록."""
        return self._feature_cols

    def _load_vif_features(self) -> list[str]:
        vif_df = pd.read_csv(self.config.vif_features_path)
        return vif_df["survived_feature"].tolist()

    def _drop_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in df.columns if c in _META_COLUMNS]
        return df.drop(columns=drop_cols, errors="ignore")

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c in _CATEGORICAL_COLUMNS if c in df.columns]
        if not cat_cols:
            return df
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)

    @staticmethod
    def _coerce_remaining_objects(df: pd.DataFrame) -> pd.DataFrame:
        """인코딩 후 남아 있는 object 컬럼을 numeric으로 강제 변환한다.

        CRP처럼 일부 행에 '음성', '<2.00' 등의 문자열이 섞여 object dtype으로 저장된
        컬럼을 pd.to_numeric으로 변환하고, 변환 불가 값은 NaN으로 처리한다.
        """
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not obj_cols:
            return df
        result = df.copy()
        for col in obj_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def _select_features(self, df: pd.DataFrame, vif_features: list[str]) -> pd.DataFrame:
        """VIF 생존 피처만 선택한다 (label 제외)."""
        available = [f for f in vif_features if f in df.columns]
        missing = [f for f in vif_features if f not in df.columns]
        if missing:
            warnings.warn(f"VIF 피처 중 데이터에 없는 컬럼: {missing}", stacklevel=2)
        return df[available]


class FoldPreprocessor:
    """Outer fold당 한 번 fit해 모든 모델이 공유하는 전처리기.

    Imputer와 StandardScaler를 fold별로 한 번만 fit하고,
    변환된 데이터를 모든 모델에 재사용한다.
    """

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.imputer_: KNNImputer | SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None

    def _build_imputer(self) -> KNNImputer | SimpleImputer:
        """설정에 따라 적절한 Imputer를 생성한다."""
        strategy = self.config.imputation_strategy
        if strategy == "knn":
            return KNNImputer(n_neighbors=self.config.knn_n_neighbors)
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=self.config.imputation_fill_value)
        return SimpleImputer(strategy=strategy)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """X_train으로 전처리기를 fit하고 변환된 X_train을 반환한다."""
        self.imputer_ = self._build_imputer()
        X_imp = self.imputer_.fit_transform(X_train)
        if self.config.scale_features:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X_imp)
        return X_imp

    def transform(self, X: np.ndarray) -> np.ndarray:
        """fit된 전처리기로 X를 변환한다."""
        assert self.imputer_ is not None, "fit_transform을 먼저 호출해야 합니다."
        X_imp = self.imputer_.transform(X)
        if self.scaler_ is not None:
            return self.scaler_.transform(X_imp)
        return X_imp


class ModelBuilder:
    """sklearn Pipeline 기반 모델 빌더."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config

    def build_classifier_pipeline(self, model_name: ModelName, scale_pos_weight: float = 1.0) -> Pipeline:
        """분류기만 포함한 단일 단계 Pipeline을 반환한다.

        전처리(KNNImputer, StandardScaler)는 FoldPreprocessor가 fold 단위로 처리하므로
        pipeline에는 classifier 단계만 포함한다.

        Args:
            model_name: 모델 종류.
            scale_pos_weight: XGBoost 전용. neg_count / pos_count 비율을 fold마다 전달한다.
        """
        return Pipeline([("classifier", self._make_classifier(model_name, scale_pos_weight))])

    def _make_classifier(self, model_name: ModelName, scale_pos_weight: float = 1.0):
        rs = self.config.random_state
        if model_name == "svm":
            return SVC(probability=True, random_state=rs, class_weight="balanced")
        if model_name == "lr":
            return LogisticRegression(random_state=rs, class_weight="balanced")
        if model_name == "rf":
            return RandomForestClassifier(random_state=rs, class_weight="balanced")
        if model_name == "xgb":
            return XGBClassifier(
                random_state=rs,
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
                eval_metric="logloss",
            )
        if model_name == "lgbm":
            return lgb.LGBMClassifier(
                random_state=rs,
                class_weight="balanced",
                verbose=-1,
            )
        if model_name == "gb":
            return GradientBoostingClassifier(random_state=rs)
        if model_name == "et":
            return ExtraTreesClassifier(random_state=rs, class_weight="balanced")
        raise ValueError(f"지원하지 않는 모델: {model_name}")


class MetricsCalculator:
    """이진 분류 평가 지표 계산."""

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        fold: int,
        model: ModelName,
        best_params: dict,
    ) -> FoldMetrics:
        """혼동행렬 기반 지표를 계산한다."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        pr_auc_score = average_precision_score(y_true, y_prob)
        return FoldMetrics(
            fold=fold,
            model=model,
            accuracy=accuracy_score(y_true, y_pred),
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            auc=auc_score,
            pr_auc=pr_auc_score,
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            best_params=best_params,
        )


class NestedCVTrainer:
    """Nested 5-Fold CV 학습·평가."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.preparator = DataPreparator(config)
        self.builder = ModelBuilder(config)
        self.calculator = MetricsCalculator()
        self._results: dict[ModelName, ModelResult] = {}
        self._X: pd.DataFrame | None = None
        self._y: pd.Series | None = None

    def prepare_data(self) -> None:
        """데이터 로딩 및 인코딩을 수행한다."""
        self._X, self._y = self.preparator.load_and_prepare()
        print(f"X shape: {self._X.shape}, y distribution: {self._y.value_counts().to_dict()}")
        print(f"선택된 피처 ({len(self.preparator.feature_cols)}개): {self.preparator.feature_cols}")

    def run(self) -> dict[ModelName, ModelResult]:
        """모든 모델에 대해 Nested CV를 수행하고 결과를 반환한다."""
        if self._X is None or self._y is None:
            self.prepare_data()

        X = self._X.values
        y = self._y.values

        outer_cv = StratifiedKFold(
            n_splits=self.config.n_outer_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        inner_cv = StratifiedKFold(
            n_splits=self.config.n_inner_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # model별 결과 버퍼 초기화
        fold_metrics_buf: dict[ModelName, list[FoldMetrics]] = {m: [] for m in self.config.models}
        fpr_buf: dict[ModelName, list[np.ndarray]] = {m: [] for m in self.config.models}
        tpr_buf: dict[ModelName, list[np.ndarray]] = {m: [] for m in self.config.models}
        precision_buf: dict[ModelName, list[np.ndarray]] = {m: [] for m in self.config.models}
        recall_buf: dict[ModelName, list[np.ndarray]] = {m: [] for m in self.config.models}

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"\n{'='*60}  Fold {fold_idx}")

            # ── 전처리: fold당 한 번만 fit ──────────────────────────
            preprocessor = FoldPreprocessor(self.config)
            X_train_proc = preprocessor.fit_transform(X[train_idx])
            X_test_proc  = preprocessor.transform(X[test_idx])
            y_train = y[train_idx]
            y_test  = y[test_idx]

            # XGBoost scale_pos_weight: neg / pos (fold train 기준)
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            spw = neg_count / pos_count if pos_count > 0 else 1.0

            # ── 모델별 하이퍼파라미터 탐색 ──────────────────────────
            for model_name in self.config.models:
                pipeline = self.builder.build_classifier_pipeline(model_name, scale_pos_weight=spw)
                gs = self._build_searcher(model_name, pipeline, inner_cv)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gs.fit(X_train_proc, y_train)

                best_clf  = gs.best_estimator_
                y_pred    = best_clf.predict(X_test_proc)
                y_prob    = best_clf.predict_proba(X_test_proc)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fpr_buf[model_name].append(fpr)
                tpr_buf[model_name].append(tpr)

                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                precision_buf[model_name].append(prec)
                recall_buf[model_name].append(rec)

                metrics = self.calculator.compute(
                    y_test, y_pred, y_prob, fold_idx, model_name, gs.best_params_
                )
                fold_metrics_buf[model_name].append(metrics)
                print(
                    f"  [{model_name.upper():3s}] AUC={metrics.auc:.4f}  "
                    f"PR-AUC={metrics.pr_auc:.4f}  "
                    f"F1={metrics.f1:.4f}  "
                    f"Acc={metrics.accuracy:.4f}  "
                    f"Sens={metrics.sensitivity:.4f}  "
                    f"Spec={metrics.specificity:.4f}"
                )

        for model_name in self.config.models:
            mean_fpr, mean_tpr = self._interpolate_roc(fpr_buf[model_name], tpr_buf[model_name])
            self._results[model_name] = ModelResult(
                model=model_name,
                fold_metrics=fold_metrics_buf[model_name],
                fpr_list=fpr_buf[model_name],
                tpr_list=tpr_buf[model_name],
                mean_fpr=mean_fpr,
                mean_tpr=mean_tpr,
                precision_list=precision_buf[model_name],
                recall_list=recall_buf[model_name],
            )

        return self._results

    @property
    def results(self) -> dict[ModelName, ModelResult]:
        """실행 결과."""
        return self._results

    def _build_searcher(
        self,
        model_name: ModelName,
        pipeline: Pipeline,
        inner_cv: StratifiedKFold,
    ) -> GridSearchCV | BayesSearchCV:
        """설정에 따라 GridSearchCV 또는 BayesSearchCV를 반환한다."""
        if self.config.tuner == "bayes":
            return BayesSearchCV(
                pipeline,
                _MODEL_BAYES_SPACES[model_name],
                n_iter=self.config.n_bayes_iter,
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1,
                refit=True,
                random_state=self.config.random_state,
            )
        return GridSearchCV(
            pipeline,
            _MODEL_PARAM_GRIDS[model_name],
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )

    @staticmethod
    def _interpolate_roc(
        fpr_list: list[np.ndarray],
        tpr_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold별 ROC를 보간하여 평균 FPR/TPR을 반환한다."""
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        return mean_fpr, mean_tpr


class ResultExporter:
    """학습 결과 시각화 및 저장."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics_csv(self, results: dict[ModelName, ModelResult], filename: str = "cv_metrics.csv") -> Path:
        """Fold별 지표를 CSV로 저장한다."""
        dfs = [r.metrics_df() for r in results.values()]
        df_all = pd.concat(dfs, ignore_index=True)
        path = self.output_dir / filename
        df_all.to_csv(path, index=False)
        print(f"Saved metrics: {path}")
        return path

    def save_summary_csv(self, results: dict[ModelName, ModelResult], filename: str = "cv_summary.csv") -> Path:
        """모델별 평균±std 요약을 CSV로 저장한다."""
        rows = []
        for model_name, result in results.items():
            summary = result.summary()
            summary.insert(0, "model", model_name)
            rows.append(summary)
        df_all = pd.concat(rows, ignore_index=True)
        path = self.output_dir / filename
        df_all.to_csv(path, index=False)
        print(f"Saved summary: {path}")
        return path

    def save_confusion_matrices(
        self,
        results: dict[ModelName, ModelResult],
        prefix: str = "cm",
    ) -> list[Path]:
        """각 모델의 Fold별 혼동행렬을 저장한다."""
        saved: list[Path] = []
        model_labels = {
            "svm": "SVM",
            "lr": "Logistic Regression",
            "rf": "Random Forest",
            "xgb": "XGBoost",
            "lgbm": "LightGBM",
            "gb": "Gradient Boosting",
            "et": "Extra Trees",
        }
        for model_name, result in results.items():
            n_folds = len(result.fold_metrics)
            fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4))
            if n_folds == 1:
                axes = [axes]
            fig.suptitle(f"{model_labels.get(model_name, model_name)} — Confusion Matrix per Fold", fontsize=13)

            for ax, fm in zip(axes, result.fold_metrics):
                cm_arr = np.array([[fm.tn, fm.fp], [fm.fn, fm.tp]])
                disp = ConfusionMatrixDisplay(cm_arr, display_labels=["Negative", "Positive"])
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(
                    f"Fold {fm.fold}\nAcc={fm.accuracy:.3f}",
                    fontsize=10,
                )

            plt.tight_layout()
            path = self.output_dir / f"{prefix}_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"Saved confusion matrix: {path}")
        return saved

    def save_roc_curves(
        self,
        results: dict[ModelName, ModelResult],
        filename: str = "roc_curves.png",
    ) -> Path:
        """모델별 Fold ROC 및 평균 ROC를 하나의 그림으로 저장한다."""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        axes = axes[0]

        model_colors = {
            "svm": "#e07b54", "lr": "#5b8db8", "rf": "#6dbf7e", "xgb": "#a97fc7",
            "lgbm": "#f0c040", "gb": "#4db8b8", "et": "#c77dbd",
        }
        model_labels = {
            "svm": "SVM", "lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
            "lgbm": "LightGBM", "gb": "Gradient Boosting", "et": "Extra Trees",
        }

        for ax, (model_name, result) in zip(axes, results.items()):
            color = model_colors.get(model_name, "steelblue")
            label = model_labels.get(model_name, model_name)

            mean_auc = np.mean([fm.auc for fm in result.fold_metrics])
            std_auc = np.std([fm.auc for fm in result.fold_metrics])

            for fold_idx, (fpr, tpr) in enumerate(zip(result.fpr_list, result.tpr_list), start=1):
                ax.plot(fpr, tpr, alpha=0.3, linewidth=1, color=color)

            ax.plot(
                result.mean_fpr,
                result.mean_tpr,
                color=color,
                linewidth=2.5,
                label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})",
            )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_title(f"{label}\nROC Curve (Nested 5-Fold CV)", fontsize=12)
            ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ROC curves: {path}")
        return path

    def save_roc_curves_individual(
        self,
        results: dict[ModelName, ModelResult],
        prefix: str = "roc",
    ) -> list[Path]:
        """모델별 개별 ROC curve 파일로 저장한다."""
        saved: list[Path] = []
        model_colors = {
            "svm": "#e07b54", "lr": "#5b8db8", "rf": "#6dbf7e", "xgb": "#a97fc7",
            "lgbm": "#f0c040", "gb": "#4db8b8", "et": "#c77dbd",
        }
        model_labels = {
            "svm": "SVM", "lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
            "lgbm": "LightGBM", "gb": "Gradient Boosting", "et": "Extra Trees",
        }

        for model_name, result in results.items():
            color = model_colors.get(model_name, "steelblue")
            label = model_labels.get(model_name, model_name)
            mean_auc = np.mean([fm.auc for fm in result.fold_metrics])
            std_auc = np.std([fm.auc for fm in result.fold_metrics])

            fig, ax = plt.subplots(figsize=(6, 5))
            for fold_idx, (fpr, tpr) in enumerate(zip(result.fpr_list, result.tpr_list), start=1):
                fold_auc = result.fold_metrics[fold_idx - 1].auc
                ax.plot(fpr, tpr, alpha=0.4, linewidth=1.2, color=color, label=f"Fold {fold_idx} (AUC={fold_auc:.3f})")

            ax.plot(
                result.mean_fpr,
                result.mean_tpr,
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=f"Mean (AUC={mean_auc:.4f} ± {std_auc:.4f})",
            )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"{label} — ROC Curve (Nested 5-Fold CV)", fontsize=13)
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()

            path = self.output_dir / f"{prefix}_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"Saved ROC curve: {path}")
        return saved

    def save_pr_curves(
        self,
        results: dict[ModelName, ModelResult],
        filename: str = "pr_curves.png",
    ) -> Path:
        """모델별 Fold PR curve 및 평균을 하나의 그림으로 저장한다."""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        axes = axes[0]

        model_colors = {
            "svm": "#e07b54", "lr": "#5b8db8", "rf": "#6dbf7e", "xgb": "#a97fc7",
            "lgbm": "#f0c040", "gb": "#4db8b8", "et": "#c77dbd",
        }
        model_labels = {
            "svm": "SVM", "lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
            "lgbm": "LightGBM", "gb": "Gradient Boosting", "et": "Extra Trees",
        }

        for ax, (model_name, result) in zip(axes, results.items()):
            color = model_colors.get(model_name, "steelblue")
            label = model_labels.get(model_name, model_name)
            mean_pr_auc = np.mean([fm.pr_auc for fm in result.fold_metrics])
            std_pr_auc = np.std([fm.pr_auc for fm in result.fold_metrics])

            for prec, rec in zip(result.precision_list, result.recall_list):
                ax.plot(rec, prec, alpha=0.3, linewidth=1, color=color)

            # 평균 PR curve: recall 그리드에서 각 fold 보간 후 평균
            mean_recall_grid = np.linspace(0, 1, 200)
            interp_precs = []
            for prec, rec in zip(result.precision_list, result.recall_list):
                interp_precs.append(np.interp(mean_recall_grid, rec[::-1], prec[::-1]))
            mean_prec = np.mean(interp_precs, axis=0)

            ax.plot(
                mean_recall_grid,
                mean_prec,
                color=color,
                linewidth=2.5,
                label=f"Mean PR (AP = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})",
            )
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("Recall", fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_title(f"{label}\nPR Curve (Nested 5-Fold CV)", fontsize=12)
            ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PR curves: {path}")
        return path

    def save_pr_curves_individual(
        self,
        results: dict[ModelName, ModelResult],
        prefix: str = "pr",
    ) -> list[Path]:
        """모델별 개별 PR curve 파일로 저장한다."""
        saved: list[Path] = []
        model_colors = {
            "svm": "#e07b54", "lr": "#5b8db8", "rf": "#6dbf7e", "xgb": "#a97fc7",
            "lgbm": "#f0c040", "gb": "#4db8b8", "et": "#c77dbd",
        }
        model_labels = {
            "svm": "SVM", "lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
            "lgbm": "LightGBM", "gb": "Gradient Boosting", "et": "Extra Trees",
        }

        for model_name, result in results.items():
            color = model_colors.get(model_name, "steelblue")
            label = model_labels.get(model_name, model_name)
            mean_pr_auc = np.mean([fm.pr_auc for fm in result.fold_metrics])
            std_pr_auc = np.std([fm.pr_auc for fm in result.fold_metrics])

            fig, ax = plt.subplots(figsize=(6, 5))
            for fold_idx, (prec, rec) in enumerate(zip(result.precision_list, result.recall_list), start=1):
                fold_pr_auc = result.fold_metrics[fold_idx - 1].pr_auc
                ax.plot(rec, prec, alpha=0.4, linewidth=1.2, color=color, label=f"Fold {fold_idx} (AP={fold_pr_auc:.3f})")

            mean_recall_grid = np.linspace(0, 1, 200)
            interp_precs = []
            for prec, rec in zip(result.precision_list, result.recall_list):
                interp_precs.append(np.interp(mean_recall_grid, rec[::-1], prec[::-1]))
            mean_prec = np.mean(interp_precs, axis=0)

            ax.plot(
                mean_recall_grid,
                mean_prec,
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=f"Mean (AP={mean_pr_auc:.4f} ± {std_pr_auc:.4f})",
            )
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title(f"{label} — PR Curve (Nested 5-Fold CV)", fontsize=13)
            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()

            path = self.output_dir / f"{prefix}_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"Saved PR curve: {path}")
        return saved

    def print_summary_table(self, results: dict[ModelName, ModelResult]) -> None:
        """콘솔/노트북에 전체 요약 테이블을 출력한다."""
        rows = []
        for model_name, result in results.items():
            df = result.metrics_df()
            row = {"model": model_name.upper()}
            for col in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "pr_auc"]:
                row[col] = f"{df[col].mean():.4f} ± {df[col].std():.4f}"
            rows.append(row)
        summary_df = pd.DataFrame(rows).set_index("model")
        try:
            from IPython.display import display
            display(summary_df)
        except ImportError:
            print(summary_df.to_string())


class MLPipeline:
    """전체 ML 실험 파이프라인 진입점."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config
        self.trainer = NestedCVTrainer(config)
        self.exporter = ResultExporter(config.output_dir)

    def run(self) -> dict[ModelName, ModelResult]:
        """데이터 준비 → 학습 → 평가 → 저장 전체 파이프라인을 실행한다."""
        tuner_info = (
            f"BayesSearchCV (n_iter={self.config.n_bayes_iter})"
            if self.config.tuner == "bayes"
            else "GridSearchCV"
        )
        strategy = self.config.imputation_strategy
        if strategy == "knn":
            imputation_info = f"KNN (k={self.config.knn_n_neighbors})"
        elif strategy == "constant":
            imputation_info = f"constant (fill={self.config.imputation_fill_value})"
        else:
            imputation_info = strategy

        print(f"{'='*60}")
        print(f"ML Pipeline: {self.config.dataset_path.name}")
        print(f"Output dir : {self.config.output_dir}")
        print(f"Imputation : {imputation_info}")
        print(f"Models     : {self.config.models}")
        print(f"Tuner      : {tuner_info}")
        print(f"{'='*60}")

        self.trainer.prepare_data()
        results = self.trainer.run()

        self.exporter.save_metrics_csv(results)
        self.exporter.save_summary_csv(results)
        self.exporter.save_confusion_matrices(results)
        self.exporter.save_roc_curves(results)
        self.exporter.save_roc_curves_individual(results)
        self.exporter.save_pr_curves(results)
        self.exporter.save_pr_curves_individual(results)
        self.exporter.print_summary_table(results)

        return results
