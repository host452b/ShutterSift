from __future__ import annotations
import base64
import logging
from pathlib import Path

from shuttersift.config import Config
from shuttersift.engine import PhotoResult

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import openai
except ImportError:
    openai = None  # type: ignore

_PROMPT_TEMPLATE = (
    "You are a professional photographer reviewing an image. "
    "Technical scores: overall {score:.0f}/100, sharpness {sharpness:.0f}, "
    "exposure {exposure:.0f}, aesthetic {aesthetic:.0f}. "
    "In 1-2 sentences, explain whether this photo is worth keeping "
    "and identify its main strength or weakness."
)


class Explainer:
    def __init__(
        self,
        config: Config,
        gguf_path: Path | None,
        api_key_anthropic: str | None,
        api_key_openai: str | None,
    ):
        self._cfg = config
        self._gguf_path = gguf_path
        self._api_key_anthropic = api_key_anthropic
        self._api_key_openai = api_key_openai
        self._moondream_model = None

    def explain(self, photo_path: Path, result: PhotoResult) -> str:
        """Return explanation string; empty string if VLM unavailable or not review."""
        if result.decision != "review":
            return ""

        prompt = _PROMPT_TEMPLATE.format(
            score=result.score,
            sharpness=result.sub_scores.sharpness,
            exposure=result.sub_scores.exposure,
            aesthetic=result.sub_scores.aesthetic,
        )

        # Priority: local Moondream → Anthropic API → OpenAI API → skip
        if self._gguf_path and self._gguf_path.exists():
            text = self._explain_moondream(photo_path, prompt)
            if text:
                return text

        if self._api_key_anthropic and anthropic:
            text = self._explain_anthropic(photo_path, prompt)
            if text:
                return text

        if self._api_key_openai and openai:
            text = self._explain_openai(photo_path, prompt)
            if text:
                return text

        return ""

    def _img_b64(self, path: Path) -> str | None:
        """Read image and return base64-encoded JPEG bytes."""
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                return None
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.standard_b64encode(buf.tobytes()).decode()
        except Exception as e:
            logger.warning("Could not encode image %s: %s", path.name, e)
            return None

    def _explain_anthropic(self, path: Path, prompt: str) -> str:
        b64 = self._img_b64(path)
        if not b64:
            return ""
        try:
            client = anthropic.Anthropic(api_key=self._api_key_anthropic)
            msg = client.messages.create(
                model=self._cfg.api_model_anthropic,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/jpeg", "data": b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return ""

    def _explain_openai(self, path: Path, prompt: str) -> str:
        b64 = self._img_b64(path)
        if not b64:
            return ""
        try:
            client = openai.OpenAI(api_key=self._api_key_openai)
            resp = client.chat.completions.create(
                model=self._cfg.api_model_openai,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("OpenAI API error: %s", e)
            return ""

    def _explain_moondream(self, path: Path, prompt: str) -> str:
        """Run local Moondream2 inference on the image."""
        try:
            import moondream as md
        except ImportError:
            logger.warning("moondream package not installed; skipping local VLM")
            return ""
        try:
            if self._moondream_model is None:
                self._moondream_model = md.vl(model=str(self._gguf_path))
            image = md.Image.from_path(str(path))
            result = self._moondream_model.query(image, prompt)
            # moondream.query returns {"answer": "..."} or a string depending on version
            if isinstance(result, dict):
                return result.get("answer", "").strip()
            return str(result).strip()
        except Exception as e:
            logger.warning("Moondream inference error: %s", e)
            return ""
