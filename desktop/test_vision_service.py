"""vision_service: request payload and detect-box parsing (no server needed)."""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_DESKTOP = Path(__file__).resolve().parent
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))

import vision_service as vs  # noqa: E402


def _jpeg(w=640, h=400) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class TestPayload(unittest.TestCase):
    def _post(self, content):
        return mock.patch.object(vs.requests, "post",
                                 return_value=_Resp(content))

    def test_thinking_disabled_by_default(self):
        with self._post("hi") as post:
            vs.chat([{"role": "user", "content": "x"}], model="m")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertNotIn("response_format", payload)

    def test_chat_json_enforces_schema(self):
        schema = {"type": "object",
                  "properties": {"clear": {"type": "boolean"}}}
        with self._post('{"clear": true}') as post:
            out = vs.chat_json([{"role": "user", "content": "q"}], schema,
                               images=[b"jpg"], model="m", max_tokens=50)
        self.assertEqual(out, {"clear": True})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"],
                         schema)
        self.assertEqual(payload["max_tokens"], 50)
        parts = payload["messages"][-1]["content"]
        self.assertEqual(parts[1]["type"], "image_url")

    def test_chat_json_truncated_reply_is_none(self):
        with self._post('{"clear": tr'):
            self.assertIsNone(vs.chat_json(
                [{"role": "user", "content": "q"}], {}, model="m"))

    def test_detect_scales_native_boxes_to_pixels(self):
        reply = json.dumps([
            {"bbox_2d": [0, 0, 500, 500], "label": "post"},
            {"bbox_2d": [100, 250, 1000, 1000], "label": "box"},
        ])
        with self._post(reply) as post:
            res = vs.detect(_jpeg(640, 400), model="m")
        self.assertEqual([b.label for b in res.boxes], ["post", "box"])
        self.assertEqual(res.boxes[0].bbox, (0, 0, 320, 200))
        self.assertEqual(res.boxes[1].bbox, (64, 100, 640, 400))
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"],
                         vs._DETECT_MAX_TOKENS)


class TestParseBoxes(unittest.TestCase):
    def test_legacy_pixel_format(self):
        text = '```json\n{"boxes":[{"label":"cup","bbox":[1,2,3,4],' \
               '"confidence":0.8}]}\n```'
        boxes = vs._parse_boxes(text)
        self.assertEqual(boxes[0].bbox, (1, 2, 3, 4))
        self.assertEqual(boxes[0].confidence, 0.8)

    def test_norm1000_needs_image_size(self):
        text = '[{"bbox_2d":[0,0,10,10],"label":"a"}]'
        self.assertEqual(vs._parse_boxes(text, coords="norm1000"), [])

    def test_skips_malformed_entries(self):
        text = json.dumps([
            {"bbox_2d": [0, 0, 10], "label": "short"},
            {"bbox_2d": [0, 0, 10, 10]},
            "junk",
            {"bbox_2d": [0, 0, 10, 10], "label": "ok"},
        ])
        boxes = vs._parse_boxes(text, 1000, 1000, coords="norm1000")
        self.assertEqual([b.label for b in boxes], ["ok"])

    def test_unparseable(self):
        self.assertEqual(vs._parse_boxes("no json here"), [])


if __name__ == "__main__":
    unittest.main()
