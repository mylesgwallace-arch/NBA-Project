import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class SigmoidProbabilityCalibrator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, probabilities, target):
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        self.model.fit(logits, np.asarray(target, dtype=int))
        return self

    def predict_proba(self, probabilities):
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class CalibratedProbabilityModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, features):
        base_probabilities = self.base_model.predict_proba(features)[:, 1]
        calibrated = self.calibrator.predict_proba(base_probabilities)
        return np.column_stack([1 - calibrated, calibrated])


class IsotonicProbabilityCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities, target):
        self.model.fit(np.asarray(probabilities, dtype=float), np.asarray(target, dtype=int))
        return self

    def predict_proba(self, probabilities):
        calibrated = self.model.predict(np.asarray(probabilities, dtype=float))
        return np.clip(calibrated, 1e-6, 1 - 1e-6)
