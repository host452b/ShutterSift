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

_PROMPT_TEMPLATE = """\
你是专业摄影师。请评价这张照片，重点关注：清晰度、曝光、构图、人物表情。
已有技术评分：综合分 {score:.0f}/100，清晰度 {sharpness:.0f}，曝光 {exposure:.0f}，审美 {aesthetic:.0f}。
用1-2句话说明是否值得保留，指出最主要的问题或亮点。"""


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
        self._gguf_model = None

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

        # Priority: GGUF local → Anthropic API → OpenAI API → skip
        if self._gguf_path:
            text = self._explain_gguf(photo_path, prompt)
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
            import numpy as np
            img = cv2.imread(str(path))
            if img is None:
                return None
            # Resize to max 1024px on longest side for API efficiency
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

    def _explain_gguf(self, path: Path, prompt: str) -> str:
        try:
            if self._gguf_model is None:
                from llama_cpp import Llama
                self._gguf_model = Llama(
                    model_path=str(self._gguf_path),
                    n_ctx=2048,
                    verbose=False,
                )
            b64 = self._img_b64(path)
            if not b64:
                return ""
            # llama-cpp-python multimodal (LLaVA-style) prompt
            output = self._gguf_model(
                f"USER: <image>\n{prompt}\nASSISTANT:",
                images=[base64.b64decode(b64)],
                max_tokens=150,
                stop=["USER:"],
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning("GGUF inference error: %s", e)
            return ""
