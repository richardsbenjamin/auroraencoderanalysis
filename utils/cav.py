from __future__ import annotations

import numpy as np

from auroraencoderanalysis._typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


def get_concept_res(log_model: LogisticRegression, embeddings: np.ndarray, concept_vector: np.ndarray) -> dict:
    concept_scores = np.dot(embeddings, concept_vector)
    predictions = log_model.predict(embeddings)
    probabilities = log_model.predict_proba(embeddings)
    return {
        'concept_scores': concept_scores,
        'predictions': predictions,
        'probabilities': probabilities
    }

def get_correlation_analysis(results: dict) -> dict:
    concept_scores = results['concept_scores']
    probabilities = results['probabilities']
    predictions = results['predictions']
    corr_with_prob = np.corrcoef(concept_scores, probabilities[:, 1])[0, 1]
    confidence = np.max(probabilities, axis=1)
    corr_with_confidence = np.corrcoef(concept_scores, confidence)[0, 1]
    class_0_scores = concept_scores[predictions == 0]
    class_1_scores = concept_scores[predictions == 1]
    return {
        "corr_with_prob": corr_with_prob,
        "corr_with_confidence": corr_with_confidence,
        "class_0_score_mean": class_0_scores.mean(),
        "class_1_score_mean": class_1_scores.mean(),
        "concept_separation": class_1_scores.mean() - class_0_scores.mean(),
    }

def get_model_and_concept_vector(res: dict) -> tuple:
    log_model = res["model"]
    concept_vector = log_model.coef_[0]
    concept_vector = concept_vector / np.linalg.norm(concept_vector)
    return log_model, concept_vector

