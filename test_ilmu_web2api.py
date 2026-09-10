import json
import unittest

from ilmu_web2api import MODEL_MAP, ilmu_events_to_openai, model_mapping, parse_sse


class ProtocolTests(unittest.TestCase):
    def test_model_mapping(self):
        self.assertEqual(model_mapping("ilmu-v3.1"), "auto")
        self.assertEqual(model_mapping("ilmu-v3.1-instant"), "low")
        self.assertEqual(model_mapping("ilmu-v3.1-thinking"), "high")
        with self.assertRaises(ValueError):
            model_mapping("unknown")

    def test_sse_filters_internal_events_and_preserves_usage(self):
        source = [
            ": keepalive\n\n",
            "event: trace\ndata: {\"content\":\"hidden\"}\n\n",
            "event: message\ndata: {\"delta\":{\"content\":\"Hi\"}}\n\n",
            "event: message\ndata: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
        ]
        events = list(parse_sse(source))
        chunks = list(ilmu_events_to_openai(events, "ilmu-v3.1", True))
        self.assertEqual([c["choices"][0]["delta"].get("content") for c in chunks[:-1]], ["Hi"])
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 1)

    def test_non_stream_conversion(self):
        chunks = list(ilmu_events_to_openai([("message", {"text": "a"}), ("message", {"text": "b"})], "ilmu-v3.1", False))
        self.assertEqual("".join(c["choices"][0]["delta"]["content"] for c in chunks), "ab")


if __name__ == "__main__":
    unittest.main()
