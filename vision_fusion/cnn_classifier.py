"""CNN-based marker classifier for identifying blurry STag HD17 markers.

Loads a trained MobileNetV3-Small model and classifies warped 128x128 grayscale
patches into marker IDs (0-156) or unknown (class 157).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class MarkerClassifier:
    """GPU-accelerated marker ID classifier using MobileNetV3-Small."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        num_classes: int = 158,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._threshold = confidence_threshold
        self._num_classes = num_classes

        self._model = self._load_model(model_path)
        self._model.eval()
        self._model.to(self._device)

        # ImageNet normalization constants
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)

    def _load_model(self, path: str) -> nn.Module:
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, self._num_classes),
        )
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model

    @torch.no_grad()
    def classify(self, patches: list[np.ndarray]) -> list[tuple[Optional[int], float]]:
        """Classify a batch of 128x128 grayscale patches.

        Returns list of (marker_id, confidence) tuples.
        marker_id is None if confidence < threshold or predicted as unknown.
        """
        if not patches:
            return []

        batch = self._preprocess(patches)
        logits = self._model(batch)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

        results: list[tuple[Optional[int], float]] = []
        for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
            marker_id = int(pred)
            confidence = float(conf)
            # Class 157 = unknown, or below threshold
            if marker_id >= self._num_classes - 1 or confidence < self._threshold:
                results.append((None, confidence))
            else:
                results.append((marker_id, confidence))
        return results

    def _preprocess(self, patches: list[np.ndarray]) -> torch.Tensor:
        """Convert list of grayscale patches to normalized batch tensor."""
        import cv2

        tensors = []
        for patch in patches:
            if patch.ndim == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if patch.shape != (128, 128):
                patch = cv2.resize(patch, (128, 128))
            t = torch.from_numpy(patch).float() / 255.0
            t = t.unsqueeze(0).expand(3, -1, -1)  # replicate to 3 channels
            tensors.append(t)

        batch = torch.stack(tensors).to(self._device)
        batch = (batch - self._mean) / self._std
        return batch
