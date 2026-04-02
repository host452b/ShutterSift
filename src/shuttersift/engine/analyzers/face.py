from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

_MP_LOADED = False
_face_detection = None
_face_mesh = None


def _ensure_mediapipe():
    global _MP_LOADED, _face_detection, _face_mesh
    if _MP_LOADED:
        return
    try:
        import mediapipe as mp
        _face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=10,
            min_detection_confidence=0.5,
        )
        _MP_LOADED = True
    except Exception as e:
        logger.warning("MediaPipe unavailable: %s", e)


@dataclass
class FaceResult:
    count: int = 0
    eye_open_score: float = 1.0    # 1 = fully open, 0 = closed; 1.0 when no faces
    smile_score: float = 0.0
    all_eyes_closed: bool = False
    face_quality_score: float = 75.0   # neutral when no faces, 0–100 when faces
    face_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


class FaceAnalyzer:
    def analyze(self, img: np.ndarray) -> FaceResult:
        _ensure_mediapipe()
        if not _MP_LOADED:
            return FaceResult()

        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Stage 1: detect faces
        det_result = _face_detection.process(rgb)
        if not det_result.detections:
            return FaceResult()

        h, w = img.shape[:2]
        bboxes = []
        for det in det_result.detections:
            bb = det.location_data.relative_bounding_box
            bboxes.append((bb.xmin, bb.ymin, bb.xmin + bb.width, bb.ymin + bb.height))

        # Stage 2: face mesh for blendshapes
        mesh_result = _face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            return FaceResult(
                count=len(bboxes),
                face_bboxes=bboxes,
                eye_open_score=1.0,
                face_quality_score=60.0,
            )

        blendshapes_list = [
            self._extract_blendshapes(lm) for lm in mesh_result.multi_face_landmarks
        ]
        result = self._compute_face_scores(blendshapes_list)
        result.count = len(bboxes)
        result.face_bboxes = bboxes
        return result

    def _extract_blendshapes(self, landmarks) -> dict[str, float]:
        """
        Compute eye and smile metrics from raw Face Mesh landmarks.
        Uses Eye Aspect Ratio (EAR) for eyes and mouth corner elevation for smile.
        """
        lm = landmarks.landmark

        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y])

        # Left eye EAR (indices: 362, 385, 387, 263, 373, 380)
        left_ear = _ear(pt(362), pt(385), pt(387), pt(263), pt(373), pt(380))
        # Right eye EAR (indices: 33, 160, 158, 133, 153, 144)
        right_ear = _ear(pt(33), pt(160), pt(158), pt(133), pt(153), pt(144))

        # Smile: mouth corners (61, 291) relative to lip center (13)
        left_corner_y = lm[61].y
        right_corner_y = lm[291].y
        lip_center_y = lm[13].y
        # Corners above lip center = smile
        smile = max(0.0, min(1.0, (lip_center_y - (left_corner_y + right_corner_y) / 2) * 10))

        # EAR: ~0.30 open, ~0.05 closed. Normalise: blink if EAR < 0.15
        # Convert to "blink probability": 1 = fully closed
        left_blink = max(0.0, min(1.0, 1.0 - (left_ear / 0.25)))
        right_blink = max(0.0, min(1.0, 1.0 - (right_ear / 0.25)))

        return {
            "eyeBlinkLeft": left_blink,
            "eyeBlinkRight": right_blink,
            "mouthSmileLeft": smile,
            "mouthSmileRight": smile,
            "cheekSquintLeft": 0.0,
            "cheekSquintRight": 0.0,
        }

    def _compute_face_scores(self, blendshapes_list: list[dict]) -> FaceResult:
        if not blendshapes_list:
            return FaceResult()

        eyes_open_per_face = []
        smiles = []
        for bs in blendshapes_list:
            left_blink = bs.get("eyeBlinkLeft", 0.0)
            right_blink = bs.get("eyeBlinkRight", 0.0)
            eye_open = 1.0 - max(left_blink, right_blink)
            eyes_open_per_face.append(eye_open)
            smile = (
                bs.get("mouthSmileLeft", 0.0)
                + bs.get("mouthSmileRight", 0.0)
                + bs.get("cheekSquintLeft", 0.0)
                + bs.get("cheekSquintRight", 0.0)
            ) / 4.0
            smiles.append(smile)

        min_eye_open = min(eyes_open_per_face)
        all_closed = all(e < 0.25 for e in eyes_open_per_face)
        avg_smile = sum(smiles) / len(smiles)

        # face_quality_score: 50% eye open + 30% smile + 20% neutral bonus
        eye_component = min_eye_open * 50.0
        smile_component = avg_smile * 30.0
        neutral_bonus = 20.0  # base for having faces at all
        face_quality = min(100.0, eye_component + smile_component + neutral_bonus)

        return FaceResult(
            eye_open_score=min_eye_open,
            smile_score=avg_smile,
            all_eyes_closed=all_closed,
            face_quality_score=face_quality,
        )


def _ear(p1, p2, p3, p4, p5, p6) -> float:
    """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2.0 * h + 1e-6)
