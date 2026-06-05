"""Nested 5-Fold Cross-Validation DL (MLP) Trainer for prediabetes/diabetes transitions.

각 은닉층은 Dense → BatchNorm → Activation → Dropout 순서로 구성된다.
아키텍처(은닉층 수, 뉴런 수, dropout_rate 등)는 DLConfig에 직접 지정한다.
"""
from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ImputationStrategy = Literal["knn", "mice", "mean", "median", "most_frequent", "constant"]

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


# ─── PyTorch MLP ──────────────────────────────────────────────────────────────

class _MLPBlock(nn.Module):
    """Dense → BatchNorm → Activation → Dropout 단일 블록."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation,
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class _TorchMLP(nn.Module):
    """Dense → BN → Act → Dropout 블록을 쌓은 MLP.

    마지막 레이어는 이진 분류를 위한 선형 출력 (logit) 1개.
    """

    _ACTIVATION_MAP: dict[str, type[nn.Module]] = {
        "relu":    nn.ReLU,
        "tanh":    nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: tuple[int, ...],
        activation: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        act_cls = self._ACTIVATION_MAP.get(activation, nn.ReLU)

        layers: list[nn.Module] = []
        prev = in_features
        for out in hidden_layer_sizes:
            layers.append(_MLPBlock(prev, out, act_cls(), dropout_rate))
            prev = out
        layers.append(nn.Linear(prev, 1))  # 출력 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(1)


# ─── sklearn 호환 래퍼 ────────────────────────────────────────────────────────

class TorchMLPClassifier:
    """PyTorch MLP를 sklearn 인터페이스로 감싼 이진 분류기.

    fit / predict / predict_proba 를 제공하여 Nested CV에서 투명하게 사용한다.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...],
        activation: str,
        dropout_rate: float,
        loss: str,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        early_stopping: bool,
        patience: int,
        validation_fraction: float,
        random_state: int,
        device: str,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.model_: _TorchMLP | None = None
        self.n_epochs_: int = 0
        self.best_threshold_: float = 0.5
        self.train_loss_history_: list[float] = []
        self.val_f1_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchMLPClassifier:
        """모델을 학습한다."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_samples = len(X)
        n_val = max(1, int(n_samples * self.validation_fraction)) if self.early_stopping else 0
        n_train = n_samples - n_val

        # stratified split for validation
        if self.early_stopping and n_val > 0:
            idx = np.random.permutation(n_samples)
            train_idx, val_idx = idx[:n_train], idx[n_train:]
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_tr, y_tr = X, y
            X_val, y_val = None, None

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos

        self.model_ = _TorchMLP(
            in_features=X.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.loss == "weighted_bce":
            pw = neg / pos if pos > 0 else 1.0
            print(f"    pos_weight = {neg} / {pos} = {pw:.4f}")
            pos_weight = torch.tensor([pw], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=min(self.batch_size, n_train),
            shuffle=True,
            drop_last=True,
        )

        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        best_val_f1 = -1.0
        patience_counter = 0
        best_state: dict | None = None
        best_threshold = 0.5

        self.train_loss_history_ = []
        self.val_f1_history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= n_train
            self.train_loss_history_.append(epoch_loss)

            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_t)
                    val_prob = torch.sigmoid(val_logits).cpu().numpy()

                # validation set에서 F1을 최대화하는 threshold 탐색
                thresh, val_f1 = self._best_threshold(y_val, val_prob)
                self.val_f1_history_.append(val_f1)

                if val_f1 > best_val_f1 + 1e-5:
                    best_val_f1 = val_f1
                    best_threshold = thresh
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if best_state is not None:
                            self.model_.load_state_dict(best_state)
                        self.n_epochs_ = epoch
                        self.best_threshold_ = best_threshold
                        return self

        self.n_epochs_ = self.max_epochs
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_threshold_ = best_threshold
        return self

    @staticmethod
    def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
        """Precision-Recall curve 기반으로 F1을 최대화하는 threshold와 F1 값을 반환한다."""
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        # prec/rec 마지막 원소는 threshold 없음 → [:-1] 로 맞춤
        denom = prec[:-1] + rec[:-1]
        f1_scores = np.zeros_like(denom)
        mask = denom > 0
        f1_scores[mask] = 2 * prec[:-1][mask] * rec[:-1][mask] / denom[mask]
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """클래스 확률을 반환한다. shape: (n_samples, 2)"""
        assert self.model_ is not None, "fit을 먼저 호출해야 합니다."
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_t)
            prob_pos = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """best_threshold_ 기준 클래스 예측을 반환한다."""
        return (self.predict_proba(X)[:, 1] >= self.best_threshold_).astype(int)


# ─── 설정 & 결과 데이터클래스 ─────────────────────────────────────────────────

class FeatureInspector:
    """인코딩 후 사용 가능한 피처 목록을 조회한다."""

    @staticmethod
    def show(dataset_path: Path, title: str = "사용 가능한 피처") -> list[str]:
        """인코딩 후 선택 가능한 컬럼명을 출력하고 목록을 반환한다."""
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
class DLConfig:
    """DL (MLP) 훈련 파이프라인 설정.

    아키텍처 관련 파라미터를 직접 지정한다.
    각 은닉층은 Dense → BatchNorm → Activation → Dropout 순서로 구성된다.
    """

    dataset_path: Path
    vif_features_path: Path
    output_dir: Path
    label_col: str = "label"

    # ── Nested CV ────────────────────────────────────────────────
    n_outer_folds: int = 5
    random_state: int = 42

    # ── 결측치 처리 ───────────────────────────────────────────────
    imputation_strategy: ImputationStrategy = "mean"
    """결측치 처리 전략.
    - 'knn'          : KNNImputer (knn_n_neighbors 값 사용)
    - 'mice'         : IterativeImputer — MICE (Multiple Imputation by Chained Equations)
                       피처 간 관계를 반복 회귀로 모델링하여 결측치를 예측 대체
    - 'mean'         : 각 피처의 평균으로 대체
    - 'median'       : 각 피처의 중앙값으로 대체
    - 'most_frequent': 각 피처의 최빈값으로 대체
    - 'constant'     : imputation_fill_value 값으로 일괄 대체
    """
    knn_n_neighbors: int = 5
    mice_max_iter: int = 10
    """MICE 전략 사용 시 최대 반복 횟수 (IterativeImputer max_iter)."""
    imputation_fill_value: float = 0.0

    # ── MLP 아키텍처 ──────────────────────────────────────────────
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    """은닉층 뉴런 수 튜플.
    예: (256, 128, 64) → 3개 은닉층 (각각 256, 128, 64 뉴런)
    각 은닉층은 Dense → BatchNorm → Activation → Dropout 으로 구성된다.
    """
    dropout_rate: float = 0.3
    """각 은닉층 뒤에 적용되는 Dropout 비율 (0.0 이면 비활성화)."""

    # ── 손실 함수 ─────────────────────────────────────────────────
    loss: Literal["bce", "weighted_bce"] = "weighted_bce"
    """손실 함수.
    - 'bce'          : 일반 BCEWithLogitsLoss
    - 'weighted_bce' : pos_weight = neg_count / pos_count 으로 소수 클래스에 더 높은 가중치 부여
    """

    # ── 최적화 ────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    """Adam 옵티마이저 학습률."""
    weight_decay: float = 1e-4
    """Adam 옵티마이저 L2 패널티 (가중치 감쇠)."""
    batch_size: int = 64
    """미니배치 크기."""

    # ── 학습 종료 조건 ────────────────────────────────────────────
    max_epochs: int = 200
    """최대 에폭 수."""
    early_stopping: bool = True
    """True이면 검증 손실이 개선되지 않을 때 조기 종료한다."""
    patience: int = 20
    """조기 종료 전 개선 없는 최대 에폭 수."""
    validation_fraction: float = 0.1
    """early_stopping=True일 때 훈련 데이터에서 검증셋으로 사용할 비율."""

    # ── 디바이스 & 피처 ──────────────────────────────────────────
    device: str = "cuda"
    """학습 디바이스. 'cuda' 지정 시 GPU 사용 불가능하면 자동으로 'cpu'로 전환된다."""
    selected_features: list[str] | None = None
    """직접 지정할 피처 목록. None이면 vif_features_path에서 자동 로드한다."""


@dataclass
class FoldMetrics:
    """단일 fold 평가 지표."""

    fold: int
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
    n_epochs: int
    threshold: float


@dataclass
class FoldArtifact:
    """단일 fold의 학습 완료 아티팩트."""

    fold: int
    model: _TorchMLP
    imputer: KNNImputer | IterativeImputer | SimpleImputer
    scaler: StandardScaler
    threshold: float
    feature_names: list[str]


@dataclass
class CVResult:
    """5-Fold CV 전체 결과."""

    fold_metrics: list[FoldMetrics]
    fpr_list: list[np.ndarray]
    tpr_list: list[np.ndarray]
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    precision_list: list[np.ndarray]
    recall_list: list[np.ndarray]
    train_loss_histories: list[list[float]]
    """Fold별 에폭당 train loss 히스토리."""
    val_f1_histories: list[list[float]]
    """Fold별 에폭당 validation F1 히스토리."""
    fold_artifacts: list[FoldArtifact] = field(default_factory=list)
    """Fold별 학습 완료 아티팩트 (model, imputer, scaler, threshold, feature_names)."""

    def metrics_df(self) -> pd.DataFrame:
        """fold별 지표 DataFrame."""
        return pd.DataFrame(
            [
                {
                    "fold": m.fold,
                    "accuracy": m.accuracy,
                    "sensitivity": m.sensitivity,
                    "specificity": m.specificity,
                    "precision": m.precision,
                    "f1": m.f1,
                    "auc": m.auc,
                    "pr_auc": m.pr_auc,
                    "threshold": m.threshold,
                    "n_epochs": m.n_epochs,
                }
                for m in self.fold_metrics
            ]
        )

    def summary(self) -> pd.DataFrame:
        """지표 평균 ± 표준편차 요약 (n_epochs 포함)."""
        df = self.metrics_df().drop(columns=["fold"])
        return pd.DataFrame(
            {
                "metric": df.columns,
                "mean": df.mean().values,
                "std": df.std().values,
                "mean±std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(df.mean(), df.std())],
            }
        )


# ─── 데이터 준비 ──────────────────────────────────────────────────────────────

class DataPreparator:
    """데이터 로딩 및 인코딩 전처리."""

    def __init__(self, config: DLConfig) -> None:
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
        NaN은 이후 FoldPreprocessor의 Imputer가 대체한다.
        """
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not obj_cols:
            return df
        result = df.copy()
        for col in obj_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def _select_features(self, df: pd.DataFrame, vif_features: list[str]) -> pd.DataFrame:
        available = [f for f in vif_features if f in df.columns]
        missing = [f for f in vif_features if f not in df.columns]
        if missing:
            warnings.warn(f"VIF 피처 중 데이터에 없는 컬럼: {missing}", stacklevel=2)
        return df[available]


class FoldPreprocessor:
    """Outer fold당 한 번 fit하는 Imputer + StandardScaler."""

    def __init__(self, config: DLConfig) -> None:
        self.config = config
        self.imputer_: KNNImputer | IterativeImputer | SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None

    def _build_imputer(self) -> KNNImputer | IterativeImputer | SimpleImputer:
        strategy = self.config.imputation_strategy
        if strategy == "knn":
            return KNNImputer(n_neighbors=self.config.knn_n_neighbors)
        if strategy == "mice":
            return IterativeImputer(
                max_iter=self.config.mice_max_iter,
                random_state=self.config.random_state,
            )
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value=self.config.imputation_fill_value)
        return SimpleImputer(strategy=strategy)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """X_train으로 전처리기를 fit하고 변환된 X_train을 반환한다."""
        self.imputer_ = self._build_imputer()
        X_imp = self.imputer_.fit_transform(X_train)
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(X_imp)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """fit된 전처리기로 X를 변환한다."""
        assert self.imputer_ is not None and self.scaler_ is not None
        return self.scaler_.transform(self.imputer_.transform(X))


# ─── 학습 & 평가 ──────────────────────────────────────────────────────────────

class MetricsCalculator:
    """이진 분류 평가 지표 계산."""

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        fold: int,
        n_epochs: int,
        threshold: float,
    ) -> FoldMetrics:
        """혼동행렬 기반 지표를 계산한다."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return FoldMetrics(
            fold=fold,
            accuracy=accuracy_score(y_true, y_pred),
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            auc=auc(fpr, tpr),
            pr_auc=average_precision_score(y_true, y_prob),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            n_epochs=n_epochs,
            threshold=threshold,
        )


class NestedCVTrainer:
    """5-Fold CV MLP 학습·평가.

    각 fold에서 DLConfig에 지정된 아키텍처로 PyTorch MLP를 학습한다.
    """

    def __init__(self, config: DLConfig) -> None:
        self.config = config
        self.preparator = DataPreparator(config)
        self.calculator = MetricsCalculator()
        self._result: CVResult | None = None
        self._X: pd.DataFrame | None = None
        self._y: pd.Series | None = None

    def prepare_data(self) -> None:
        """데이터 로딩 및 인코딩을 수행한다."""
        self._X, self._y = self.preparator.load_and_prepare()
        print(f"X shape: {self._X.shape}, y distribution: {self._y.value_counts().to_dict()}")
        print(f"선택된 피처 ({len(self.preparator.feature_cols)}개): {self.preparator.feature_cols}")

    def run(self) -> CVResult:
        """5-Fold CV를 수행하고 결과를 반환한다."""
        if self._X is None or self._y is None:
            self.prepare_data()

        X = self._X.values
        y = self._y.values

        outer_cv = StratifiedKFold(
            n_splits=self.config.n_outer_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        fold_metrics: list[FoldMetrics] = []
        fpr_list: list[np.ndarray] = []
        tpr_list: list[np.ndarray] = []
        precision_list: list[np.ndarray] = []
        recall_list: list[np.ndarray] = []
        train_loss_histories: list[list[float]] = []
        val_f1_histories: list[list[float]] = []
        fold_artifacts: list[FoldArtifact] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"\n{'='*60}  Fold {fold_idx}")

            preprocessor = FoldPreprocessor(self.config)
            X_train_proc = preprocessor.fit_transform(X[train_idx])
            X_test_proc  = preprocessor.transform(X[test_idx])
            y_train = y[train_idx]
            y_test  = y[test_idx]

            clf = TorchMLPClassifier(
                hidden_layer_sizes=self.config.hidden_layer_sizes,
                activation="relu",
                dropout_rate=self.config.dropout_rate,
                loss=self.config.loss,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                batch_size=self.config.batch_size,
                max_epochs=self.config.max_epochs,
                early_stopping=self.config.early_stopping,
                patience=self.config.patience,
                validation_fraction=self.config.validation_fraction,
                random_state=self.config.random_state + fold_idx,
                device=self.config.device,
            )
            clf.fit(X_train_proc, y_train)
            train_loss_histories.append(clf.train_loss_history_)
            val_f1_histories.append(clf.val_f1_history_)

            # validation F1 최적 threshold로 test set 평가
            threshold = clf.best_threshold_
            y_prob = clf.predict_proba(X_test_proc)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            precision_list.append(prec)
            recall_list.append(rec)

            metrics = self.calculator.compute(y_test, y_pred, y_prob, fold_idx, clf.n_epochs_, threshold)
            fold_metrics.append(metrics)
            print(
                f"  AUC={metrics.auc:.4f}  "
                f"PR-AUC={metrics.pr_auc:.4f}  "
                f"F1={metrics.f1:.4f}  "
                f"Acc={metrics.accuracy:.4f}  "
                f"Sens={metrics.sensitivity:.4f}  "
                f"Spec={metrics.specificity:.4f}  "
                f"(threshold={threshold:.3f}  epochs={clf.n_epochs_})"
            )

            fold_artifacts.append(FoldArtifact(
                fold=fold_idx,
                model=clf.model_,
                imputer=preprocessor.imputer_,
                scaler=preprocessor.scaler_,
                threshold=threshold,
                feature_names=self.preparator.feature_cols,
            ))

        mean_fpr, mean_tpr = self._interpolate_roc(fpr_list, tpr_list)
        self._result = CVResult(
            fold_metrics=fold_metrics,
            fpr_list=fpr_list,
            tpr_list=tpr_list,
            mean_fpr=mean_fpr,
            mean_tpr=mean_tpr,
            precision_list=precision_list,
            recall_list=recall_list,
            train_loss_histories=train_loss_histories,
            val_f1_histories=val_f1_histories,
            fold_artifacts=fold_artifacts,
        )
        return self._result

    @property
    def result(self) -> CVResult | None:
        """실행 결과."""
        return self._result

    @staticmethod
    def _interpolate_roc(
        fpr_list: list[np.ndarray],
        tpr_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        return mean_fpr, mean_tpr


# ─── 결과 저장 ────────────────────────────────────────────────────────────────

class ResultExporter:
    """학습 결과 시각화 및 저장."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.eval_dir = output_dir / "evaluation"
        self.model_dir = output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_artifacts(self, result: CVResult, config: "DLConfig") -> Path:
        """Fold별 모델·전처리기·threshold 및 공통 설정을 models/ 폴더에 저장한다.

        저장 구조::

            models/
            ├── fold1_model.pt        — PyTorch state_dict
            ├── fold1_imputer.pkl     — fitted Imputer
            ├── fold1_scaler.pkl      — fitted StandardScaler
            ├── fold1_threshold.json  — optimal threshold
            ├── fold2_model.pt
            ├── ...
            └── model_config.json     — 아키텍처·피처·설정 정보
        """
        for artifact in result.fold_artifacts:
            fold = artifact.fold
            # 모델 가중치
            torch.save(
                artifact.model.state_dict(),
                self.model_dir / f"fold{fold}_model.pt",
            )
            # Imputer
            with open(self.model_dir / f"fold{fold}_imputer.pkl", "wb") as f:
                pickle.dump(artifact.imputer, f)
            # Scaler
            with open(self.model_dir / f"fold{fold}_scaler.pkl", "wb") as f:
                pickle.dump(artifact.scaler, f)
            # Threshold
            (self.model_dir / f"fold{fold}_threshold.json").write_text(
                json.dumps({"fold": fold, "threshold": artifact.threshold}, indent=2),
                encoding="utf-8",
            )

        # 공통 설정 (아키텍처 · 피처 · 하이퍼파라미터)
        first = result.fold_artifacts[0] if result.fold_artifacts else None
        model_config: dict = {
            "architecture": {
                "hidden_layer_sizes": list(config.hidden_layer_sizes),
                "dropout_rate": config.dropout_rate,
                "activation": "relu",
            },
            "training": {
                "loss": config.loss,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "max_epochs": config.max_epochs,
                "early_stopping": config.early_stopping,
                "patience": config.patience,
                "validation_fraction": config.validation_fraction,
            },
            "imputation": {
                "strategy": config.imputation_strategy,
                "knn_n_neighbors": config.knn_n_neighbors,
                "mice_max_iter": config.mice_max_iter,
            },
            "cv": {
                "n_folds": config.n_outer_folds,
                "random_state": config.random_state,
            },
            "feature_names": first.feature_names if first else [],
            "n_features": len(first.feature_names) if first else 0,
            "fold_thresholds": {
                f"fold{a.fold}": a.threshold for a in result.fold_artifacts
            },
        }
        config_path = self.model_dir / "model_config.json"
        config_path.write_text(
            json.dumps(model_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved artifacts → {self.model_dir}")
        return self.model_dir

    def save_metrics_csv(self, result: CVResult, filename: str = "cv_metrics.csv") -> Path:
        """Fold별 지표를 CSV로 저장한다."""
        path = self.eval_dir / filename
        result.metrics_df().to_csv(path, index=False)
        print(f"Saved metrics: {path}")
        return path

    def save_summary_csv(self, result: CVResult, filename: str = "cv_summary.csv") -> Path:
        """평균±std 요약을 CSV로 저장한다."""
        path = self.eval_dir / filename
        result.summary().to_csv(path, index=False)
        print(f"Saved summary: {path}")
        return path

    def save_confusion_matrices(self, result: CVResult, prefix: str = "cm") -> list[Path]:
        """Fold별 혼동행렬을 저장한다."""
        n_folds = len(result.fold_metrics)
        fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4))
        if n_folds == 1:
            axes = [axes]
        fig.suptitle("MLP — Confusion Matrix per Fold", fontsize=13)

        for ax, fm in zip(axes, result.fold_metrics):
            cm_arr = np.array([[fm.tn, fm.fp], [fm.fn, fm.tp]])
            disp = ConfusionMatrixDisplay(cm_arr, display_labels=["Negative", "Positive"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"Fold {fm.fold}\nAcc={fm.accuracy:.3f}", fontsize=10)

        plt.tight_layout()
        path = self.eval_dir / f"{prefix}_mlp.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix: {path}")
        return [path]

    def save_roc_curve(self, result: CVResult, filename: str = "roc_mlp.png") -> Path:
        """Fold ROC 및 평균 ROC를 저장한다."""
        color = "#5b8db8"
        mean_auc = np.mean([fm.auc for fm in result.fold_metrics])
        std_auc = np.std([fm.auc for fm in result.fold_metrics])

        fig, ax = plt.subplots(figsize=(6, 5))
        for fold_idx, (fpr, tpr) in enumerate(zip(result.fpr_list, result.tpr_list), start=1):
            fold_auc = result.fold_metrics[fold_idx - 1].auc
            ax.plot(fpr, tpr, alpha=0.4, linewidth=1.2, color=color,
                    label=f"Fold {fold_idx} (AUC={fold_auc:.3f})")
        ax.plot(
            result.mean_fpr, result.mean_tpr,
            color=color, linewidth=2.5,
            label=f"Mean (AUC={mean_auc:.4f} ± {std_auc:.4f})",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("MLP — ROC Curve (5-Fold CV)", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        path = self.eval_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ROC curve: {path}")
        return path

    def save_pr_curve(self, result: CVResult, filename: str = "pr_mlp.png") -> Path:
        """Fold PR curve 및 평균을 저장한다."""
        color = "#6dbf7e"
        mean_pr_auc = np.mean([fm.pr_auc for fm in result.fold_metrics])
        std_pr_auc = np.std([fm.pr_auc for fm in result.fold_metrics])

        fig, ax = plt.subplots(figsize=(6, 5))
        for fold_idx, (prec, rec) in enumerate(zip(result.precision_list, result.recall_list), start=1):
            fold_pr_auc = result.fold_metrics[fold_idx - 1].pr_auc
            ax.plot(rec, prec, alpha=0.4, linewidth=1.2, color=color,
                    label=f"Fold {fold_idx} (AP={fold_pr_auc:.3f})")

        mean_recall_grid = np.linspace(0, 1, 200)
        mean_prec = np.mean(
            [np.interp(mean_recall_grid, rec[::-1], prec[::-1])
             for prec, rec in zip(result.precision_list, result.recall_list)],
            axis=0,
        )
        ax.plot(
            mean_recall_grid, mean_prec,
            color=color, linewidth=2.5,
            label=f"Mean (AP={mean_pr_auc:.4f} ± {std_pr_auc:.4f})",
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("MLP — PR Curve (5-Fold CV)", fontsize=13)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        path = self.eval_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PR curve: {path}")
        return path

    def save_training_history_csv(
        self,
        result: CVResult,
        filename: str = "training_history.csv",
    ) -> Path:
        """Fold별 에폭당 train loss / val F1 히스토리를 CSV로 저장한다."""
        rows = []
        for fold_idx, (losses, f1s) in enumerate(
            zip(result.train_loss_histories, result.val_f1_histories), start=1
        ):
            for epoch, train_loss in enumerate(losses, start=1):
                val_f1 = f1s[epoch - 1] if epoch <= len(f1s) else float("nan")
                rows.append({"fold": fold_idx, "epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        df = pd.DataFrame(rows)
        path = self.eval_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved training history: {path}")
        return path

    def save_learning_curves(
        self,
        result: CVResult,
        filename: str = "learning_curves.png",
    ) -> Path:
        """Fold별 학습 곡선 (train loss / val F1)을 저장한다."""
        n_folds = len(result.train_loss_histories)
        fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 8))
        if n_folds == 1:
            axes = axes.reshape(2, 1)

        colors_loss = "#e07b54"
        colors_f1   = "#5b8db8"

        for fold_idx, (losses, f1s) in enumerate(
            zip(result.train_loss_histories, result.val_f1_histories)
        ):
            epochs_loss = range(1, len(losses) + 1)
            epochs_f1   = range(1, len(f1s) + 1)
            best_epoch  = result.fold_metrics[fold_idx].n_epochs

            ax_loss = axes[0][fold_idx]
            ax_loss.plot(epochs_loss, losses, color=colors_loss, linewidth=1.5)
            ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"stop={best_epoch}")
            ax_loss.set_title(f"Fold {fold_idx + 1} — Train Loss", fontsize=11)
            ax_loss.set_xlabel("Epoch", fontsize=10)
            ax_loss.set_ylabel("Loss", fontsize=10)
            ax_loss.legend(fontsize=8)

            ax_f1 = axes[1][fold_idx]
            ax_f1.plot(epochs_f1, f1s, color=colors_f1, linewidth=1.5)
            ax_f1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"stop={best_epoch}")
            ax_f1.set_title(f"Fold {fold_idx + 1} — Val F1", fontsize=11)
            ax_f1.set_xlabel("Epoch", fontsize=10)
            ax_f1.set_ylabel("F1", fontsize=10)
            ax_f1.legend(fontsize=8)

        plt.suptitle("MLP — Learning Curves (5-Fold CV)", fontsize=13, y=1.01)
        plt.tight_layout()
        path = self.eval_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved learning curves: {path}")
        return path

    def print_summary_table(self, result: CVResult) -> None:
        """콘솔/노트북에 요약 테이블을 출력한다."""
        summary = result.summary().set_index("metric")[["mean", "std", "mean±std"]]
        try:
            from IPython.display import display
            display(summary)
        except ImportError:
            print(summary.to_string())


# ─── 파이프라인 진입점 ─────────────────────────────────────────────────────────

class DLPipeline:
    """전체 DL (MLP) 실험 파이프라인 진입점."""

    def __init__(self, config: DLConfig) -> None:
        self.config = config
        self.trainer = NestedCVTrainer(config)
        self.exporter = ResultExporter(config.output_dir)

    def run(self) -> CVResult:
        """데이터 준비 → 학습 → 평가 → 저장 전체 파이프라인을 실행한다."""
        c = self.config
        strategy = c.imputation_strategy
        imputation_info = (
            f"KNN (k={c.knn_n_neighbors})" if strategy == "knn"
            else f"MICE (max_iter={c.mice_max_iter})" if strategy == "mice"
            else f"constant (fill={c.imputation_fill_value})" if strategy == "constant"
            else strategy
        )
        device_info = "CUDA" if torch.cuda.is_available() and c.device == "cuda" else "CPU"

        print(f"{'='*60}")
        print(f"DL Pipeline    : {c.dataset_path.name}")
        print(f"Output dir     : {c.output_dir}")
        print(f"Device         : {device_info}")
        print(f"Imputation     : {imputation_info}")
        print(f"Architecture   : {c.hidden_layer_sizes}  ({len(c.hidden_layer_sizes)} hidden layers)")
        print(f"Block          : Dense → BatchNorm → ReLU → Dropout({c.dropout_rate})")
        print(f"Loss           : {c.loss}")
        print(f"Optimizer      : Adam  lr={c.learning_rate}  weight_decay={c.weight_decay}")
        print(f"Batch size     : {c.batch_size}")
        print(f"Max epochs     : {c.max_epochs}  early_stopping={c.early_stopping}  patience={c.patience}")
        print(f"{'='*60}")

        self.trainer.prepare_data()
        result = self.trainer.run()

        self.exporter.save_artifacts(result, c)
        self.exporter.save_metrics_csv(result)
        self.exporter.save_summary_csv(result)
        self.exporter.save_training_history_csv(result)
        self.exporter.save_learning_curves(result)
        self.exporter.save_confusion_matrices(result)
        self.exporter.save_roc_curve(result)
        self.exporter.save_pr_curve(result)

        return result
