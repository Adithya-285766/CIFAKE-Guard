"""CIFAKE-Guard: detection of AI-generated images (CIFAKE benchmark).

Label convention used everywhere in this package: 1 = FAKE (AI-generated, positive class), 0 = REAL.
Every model consumes float tensors of shape (B, 3, 32, 32) in [0, 1] and returns one raw logit per image;
model-specific resizing / normalisation happens inside the model so that all models share one input
contract (needed for ensembling, explainability and the API).
"""

__version__ = "2.0.0"

FAKE_LABEL = 1
REAL_LABEL = 0
CLASS_NAMES = ("REAL", "FAKE")
