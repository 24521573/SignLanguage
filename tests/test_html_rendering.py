import importlib.util
import sys
import types
import unittest
from functools import cache
from pathlib import Path
from unittest.mock import patch


DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from html_rendering import escape_html


@cache
def load_app_module():
    cv2 = types.ModuleType("cv2")
    cv2.data = types.SimpleNamespace(haarcascades=".")
    cv2.CascadeClassifier = lambda _path: object()

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda _name: "cpu"
    torch.inference_mode = lambda: lambda function: function

    transformers = types.ModuleType("transformers")
    transformers.VideoMAEForVideoClassification = object

    stubs = {
        "cv2": cv2,
        "gradio": types.ModuleType("gradio"),
        "imageio_ffmpeg": types.ModuleType("imageio_ffmpeg"),
        "torch": torch,
        "transformers": transformers,
    }
    spec = importlib.util.spec_from_file_location("sign_language_app", DEMO_DIR / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class EscapeHtmlTests(unittest.TestCase):
    def test_escapes_markup_in_dynamic_content(self) -> None:
        self.assertEqual(
            escape_html('<script>alert("unsafe")</script>'),
            "&lt;script&gt;alert(&quot;unsafe&quot;)&lt;/script&gt;",
        )

    def test_prediction_card_renders_label_as_text(self) -> None:
        rendered = load_app_module().prediction_card("<strong>unsafe</strong>", 90.0)

        self.assertIn("&lt;strong&gt;unsafe&lt;/strong&gt;", rendered)
        self.assertNotIn("<strong>unsafe</strong>", rendered)

    def test_error_card_renders_message_as_text(self) -> None:
        rendered = load_app_module().error_prediction("<img src=x onerror=alert(1)>")

        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", rendered)
        self.assertNotIn("<img src=x onerror=alert(1)>", rendered)


if __name__ == "__main__":
    unittest.main()
