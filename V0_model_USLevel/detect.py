# detect_v1.py
# --- Advanced anomaly detection using PyCaret and PyOD with ensemble modeling ---
# This replaces the traditional V0 anomaly detection with ML-based approaches
import logging
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from pycaret.anomaly import (
    setup as anom_setup,
    create_model as anom_create,
    predict_model as anom_predict,
)

# Optional: PyOD fallbacks for models not exposed by current PyCaret build
try:
    from pyod.models.hbos import HBOS
    from pyod.models.copod import COPOD
    from pyod.models.ocsvm import OCSVM
    from pyod.models.cblof import CBLOF
    PYOD_AVAILABLE = True
except Exception:
    HBOS = COPOD = OCSVM = CBLOF = None
    PYOD_AVAILABLE = False

RANDOM_SEED = 42

class AnomalyDetectorV1:
    def __init__(
        self,
        models=("iforest", "hbos", "copod", "knn", "svm", "ocsvm", "cblof"),
        alert_budget_per_week=2,
        cooldown_days=1,
        quantile_bounds=(0.95, 0.999),
        verbose=False,
    ):
        self.models = list(models)
        self.alert_budget_per_week = int(alert_budget_per_week) if alert_budget_per_week is not None else 0
        self.cooldown_days = int(cooldown_days) if cooldown_days is not None else 0
        self.qmin, self.qmax = float(quantile_bounds[0]), float(quantile_bounds[1])
        self.verbose = verbose

        self.fitted_models_ = {}
        self.feature_cols_ = None
        self.pyod_feature_medians_ = None
        self.logger = logging.getLogger(__name__)
        if self.verbose and not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    # ---------- Feature engineering ----------
    @staticmethod
    def _add_roll_feats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["mean_7"]  = g["job_count"].rolling(7,  min_periods=3).mean()
        g["mean_14"] = g["job_count"].rolling(14, min_periods=5).mean()
        g["mean_28"] = g["job_count"].rolling(28, min_periods=7).mean()
        g["std_7"]   = g["job_count"].rolling(7,  min_periods=3).std(ddof=0)
        g["std_28"]  = g["job_count"].rolling(28, min_periods=7).std(ddof=0)
        g["mad_28"] = g["job_count"].rolling(28, min_periods=7).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=False
        )
        g["pct_chg_1"] = g["job_count"].pct_change(1)
        g["pct_chg_7"] = g["job_count"].pct_change(7)
        g["z_7"]  = (g["job_count"] - g["mean_7"])  / (g["std_7"].replace(0, np.nan))
        g["z_28"] = (g["job_count"] - g["mean_28"]) / (g["std_28"].replace(0, np.nan))
        g["roll_min_14"] = g["job_count"].rolling(14, min_periods=5).min()
        g["roll_max_14"] = g["job_count"].rolling(14, min_periods=5).max()
        g["lt_mean"] = g["job_count"].expanding(min_periods=14).mean()
        g["share_vs_lt"] = g["job_count"] / (g["lt_mean"].replace(0, np.nan))
        return g

    @staticmethod
    def _build_features(df_hist: pd.DataFrame):
        df = df_hist.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["src", "date"])
        df["dow"] = df["date"].dt.dayofweek
        df["woy"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df = df.groupby(["geo_level", "src"], group_keys=False).apply(
            AnomalyDetectorV1._add_roll_feats
        )
        feature_cols = [
            "job_count", "mean_7", "mean_14", "mean_28",
            "std_7", "std_28", "mad_28",
            "pct_chg_1", "pct_chg_7",
            "z_7", "z_28",
            "roll_min_14", "roll_max_14",
            "lt_mean", "share_vs_lt",
            "dow", "woy", "month",
        ]
        df_feat = df.dropna(subset=["mean_7", "mean_14", "mean_28"]).reset_index(drop=True)
        return df_feat, feature_cols

    # ---------- Modeling ----------
    def _train_models(self, df_feat: pd.DataFrame, feature_cols):
        # Initialize PyCaret session for models it supports
        _ = anom_setup(
            data=df_feat[feature_cols],
            session_id=RANDOM_SEED,
            verbose=False,
        )
        fitted = {}
        X = df_feat[feature_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.pyod_feature_medians_ = X.median(numeric_only=True)

        for m in self.models:
            m_lower = str(m).lower()
            # First, try PyCaret-native models
            try:
                if m_lower in {"iforest", "knn", "svm"}:
                    mdl = anom_create(m_lower)
                    fitted[m_lower] = ("pycaret", mdl)
                    self.logger.info(f"Trained model: {m_lower}")
                    continue
            except Exception as e:
                self.logger.error(f"Failed to create/train model '{m_lower}' via PyCaret: {e}")

            # Fallback to PyOD for models not available in current PyCaret build
            try:
                if not PYOD_AVAILABLE:
                    raise RuntimeError("PyOD not available for fallbacks")

                X_pyod = X.fillna(self.pyod_feature_medians_)
                if m_lower == "hbos" and HBOS is not None:
                    mdl = HBOS()
                    mdl.fit(X_pyod)
                    fitted[m_lower] = ("pyod", mdl)
                    self.logger.info("Trained model via PyOD: hbos")
                elif m_lower == "copod" and COPOD is not None:
                    mdl = COPOD()
                    mdl.fit(X_pyod)
                    fitted[m_lower] = ("pyod", mdl)
                    self.logger.info("Trained model via PyOD: copod")
                elif m_lower in {"ocsvm", "oneclasssvm"} and OCSVM is not None:
                    mdl = OCSVM()
                    mdl.fit(X_pyod)
                    fitted[m_lower] = ("pyod", mdl)
                    self.logger.info("Trained model via PyOD: ocsvm")
                elif m_lower == "cblof" and CBLOF is not None:
                    mdl = CBLOF(check_estimator=False)
                    mdl.fit(X_pyod)
                    fitted[m_lower] = ("pyod", mdl)
                    self.logger.info("Trained model via PyOD: cblof")
                else:
                    self.logger.error(f"Model '{m}' not recognized or unsupported by PyCaret/PyOD")
            except Exception as e:
                self.logger.error(f"Failed to create/train model '{m_lower}' via PyOD: {e}")

        self.fitted_models_ = fitted
        self.feature_cols_ = list(feature_cols)
        return fitted

    @staticmethod
    def _minmax_norm(df_scores: pd.DataFrame) -> pd.DataFrame:
        out = df_scores.copy()
        for c in out.columns:
            vals = out[c].astype(float)
            cmin, cmax = np.nanmin(vals), np.nanmax(vals)
            out[c] = (vals - cmin) / (cmax - cmin) if np.isfinite(cmax - cmin) and (cmax - cmin) > 0 else 0.0
        return out

    @staticmethod
    def _align_direction(score: np.ndarray, label_series) -> np.ndarray:
        if label_series is None:
            return score
        try:
            lab = pd.Series(label_series).astype(int)
            if lab.nunique() != 2:
                return score
            mean_anom = np.nanmean(score[lab == 1])
            mean_norm = np.nanmean(score[lab == 0])
            if np.isfinite(mean_anom) and np.isfinite(mean_norm) and mean_anom < mean_norm:
                return -score
        except Exception:
            pass
        return score

    def _score(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        X = df_feat[self.feature_cols_].copy()
        # Clean features for scoring paths that don't handle NaNs/infs
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        scores = pd.DataFrame(index=df_feat.index)
        for m, model_info in self.fitted_models_.items():
            try:
                model_type, mdl = model_info
                if model_type == "pycaret":
                    pred = anom_predict(mdl, data=X_clean)
                    s = pd.to_numeric(pred.get("Anomaly_Score"), errors="coerce").values
                    label = pred["Anomaly"] if "Anomaly" in pred.columns else None
                    s = self._align_direction(s, label)
                elif model_type == "pyod":
                    # PyOD: higher decision_function scores = more anomalous
                    X_pyod = X_clean.fillna(self.pyod_feature_medians_)
                    s = mdl.decision_function(X_pyod)
                    s = np.asarray(s, dtype=float)
                else:
                    raise RuntimeError(f"Unknown model type for '{m}': {model_type}")
                scores[f"score_{m}"] = s
            except Exception as e:
                self.logger.error(f"Scoring failure for model '{m}': {e}")
                scores[f"score_{m}"] = np.nan
        scores = scores.apply(lambda col: col.fillna(np.nanmedian(col.values)))
        scores = self._minmax_norm(scores)
        scores["score_ens"] = scores.mean(axis=1)
        return scores

    # ---------- Thresholding & cooldown ----------
    def _calibrate_threshold_for_group(self, g: pd.DataFrame, score_col: str) -> float:
        # Use tunable lower quantile directly from configuration for sensitivity control
        dates = pd.to_datetime(g["date"])
        days = int((dates.max() - dates.min()).days) + 1
        
        q = float(self.qmin)
        threshold = float(np.quantile(g[score_col].values, q))
        
        if self.verbose:
            self.logger.info(f"Threshold calibration: {len(g)} records, {days} days, quantile={q:.3f}, threshold={threshold:.3f}")
        
        return threshold

    @staticmethod
    def _apply_cooldown(df_out: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
        if cooldown_days <= 0:
            return df_out
        def _cool(group):
            group = group.sort_values("date").copy()
            last_alert_day = None
            cooled = []
            for _, row in group.iterrows():
                is_alert = int(row["is_anomaly"])
                cur_day = pd.to_datetime(row["date"]).date()
                keep = is_alert
                if is_alert and last_alert_day is not None:
                    if (pd.Timestamp(cur_day) - pd.Timestamp(last_alert_day)).days <= cooldown_days:
                        keep = 0
                if is_alert and keep == 1:
                    last_alert_day = cur_day
                cooled.append(keep)
            group["is_anomaly"] = cooled
            return group
        return df_out.groupby(["geo_level", "src"], group_keys=False).apply(_cool).reset_index(drop=True)

    # ---------- Public API ----------
    def detect_anomalies(self, df_hist: pd.DataFrame, csv_path: str = None):
        """
        Detect anomalies using ensemble ML models
        
        Args:
            df_hist: DataFrame with columns ['date', 'geo_level', 'src', 'job_count']
                     Expected format from the preprocessing pipeline
            csv_path: Optional path to save results
            
        Returns:
            df_out: DataFrame with anomaly scores and predictions
            pivot: Pivot table of job counts by segment and date
            summary: Dictionary with detection statistics
        """
        if df_hist is None or len(df_hist) == 0:
            self.logger.error("Empty history dataframe received.")
            return pd.DataFrame(), pd.DataFrame(), {}
        df_feat, feature_cols = self._build_features(df_hist)
        if len(df_feat) == 0:
            self.logger.error("No rows available after feature warmup drop. Need more history.")
            return pd.DataFrame(), pd.DataFrame(), {}
        self._train_models(df_feat, feature_cols)
        df_scores = self._score(df_feat)

        cols_keep = ["geo_level", "src", "date", "job_count"]
        df_out = pd.concat([df_feat[cols_keep].reset_index(drop=True), df_scores.reset_index(drop=True)], axis=1)
        df_out["date_only"] = pd.to_datetime(df_out["date"]).dt.date

        thresholds = (
            df_out.groupby(["geo_level", "src"], group_keys=False)
                  .apply(lambda g: pd.Series({"threshold": self._calibrate_threshold_for_group(g, "score_ens")}))
                  .reset_index()
        )
        df_out = df_out.merge(thresholds, on=["geo_level", "src"], how="left")
        df_out["is_anomaly"] = (df_out["score_ens"] >= df_out["threshold"]).astype(int)
        df_out = self._apply_cooldown(df_out, self.cooldown_days)
        
        # Sort by grouping fields (geo_level, src) then by date for logical organization
        df_out = df_out.sort_values(["geo_level", "src", "date"]).reset_index(drop=True)

        pivot = (df_out.pivot_table(index=["geo_level", "src"], columns="date", values="job_count").sort_index())

        summary = {
            "total_records": int(len(df_out)),
            "total_anomalies": int(df_out["is_anomaly"].sum()),
            "anomaly_rate": float(df_out["is_anomaly"].mean()),
            "date_range": f"{df_out['date'].min()} to {df_out['date'].max()}",
            "segments": int(df_out[["geo_level", "src"]].drop_duplicates().shape[0]),
            "quantile_bounds": (self.qmin, self.qmax),
            "alert_budget_per_week": self.alert_budget_per_week,
            "today_anomalies": None,
        }
        try:
            latest_day = pd.to_datetime(df_out["date"]).dt.date.max()
            mask_today = pd.to_datetime(df_out["date"]).dt.date == latest_day
            today_df = df_out.loc[mask_today & (df_out["is_anomaly"] == 1), cols_keep + ["score_ens"]]
            if len(today_df):
                summary["today_anomalies"] = today_df.sort_values(["geo_level", "src"])
        except Exception as e:
            self.logger.warning(f"Unable to compute today's anomalies: {e}")

        if csv_path:
            df_out.to_csv(csv_path, index=False)
            self.logger.info(f"v1 anomaly detection results saved to {csv_path}")

        return df_out, pivot, summary


def detect_anomalies_v1(df_hist: pd.DataFrame, csv_path: str = None, **kwargs):
    detector = AnomalyDetectorV1(**kwargs)
    return detector.detect_anomalies(df_hist, csv_path=csv_path)


# Backward compatibility for existing pipeline
class AnomalyDetector:
    @staticmethod
    def detect_anomalies(df, csv_path=None, **kwargs):
        """
        Backward compatibility wrapper for V0 pipeline
        This maintains the same interface while using V1 detection logic
        """
        detector = AnomalyDetectorV1(
            models=("iforest", "hbos", "copod", "knn", "svm", "ocsvm", "cblof"),
            alert_budget_per_week=2,
            cooldown_days=1,
            quantile_bounds=(0.95, 0.999),
            verbose=True
        )
        return detector.detect_anomalies(df, csv_path=csv_path)
