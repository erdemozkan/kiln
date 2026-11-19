"""The codec alone, and round trips through Node (skipped without Node or Kiln's build)."""
import array
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import kiln_webgpu
from kiln_webgpu import KilnError, RawImage, Tensor
from kiln_webgpu._bridge import BRIDGE, find_kiln, find_node, node_version, MIN_NODE
from kiln_webgpu._codec import decode, encode, pack, unpack
from kiln_webgpu._pipeline import Pipeline, _media


def have_node():
    try:
        return node_version(find_node()) >= MIN_NODE and bool(find_kiln())
    except KilnError:
        return False


class Codec(unittest.TestCase):
    def test_frame_round_trip(self):
        blobs, callbacks = [], []
        body = encode({"a": [1, 2.5, "x", None, True], "b": b"\x01\x02", "big": 2**60, "nan": float("nan"), "f": array.array("f", [1.5, -2.0])}, blobs, callbacks)
        frame = pack({"id": 1, "body": body}, blobs)
        header, got = unpack(io.BytesIO(frame).read)
        out = decode(header["body"], got)
        self.assertEqual(out["a"], [1, 2.5, "x", None, True])
        self.assertEqual(out["b"].data, b"\x01\x02")
        self.assertEqual(out["big"], 2**60)
        self.assertNotEqual(out["nan"], out["nan"])
        self.assertEqual(out["f"].tolist(), [1.5, -2.0])
        self.assertEqual(out["f"].dtype, "float32")

    def test_truncated_frame_is_the_end(self):
        frame = pack({"id": 1}, [b"abcdef"])
        self.assertIsNone(unpack(io.BytesIO(frame[:-2]).read))
        self.assertIsNone(unpack(io.BytesIO(b"").read))

    def test_tensor_and_image_shapes(self):
        t = decode({"type": "float32", "dims": [2, 2], "data": {"$blob": 0, "dtype": "float32"}}, [array.array("f", [1, 2, 3, 4]).tobytes()])
        self.assertIsInstance(t, Tensor)
        self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t.shape, (2, 2))
        m = decode({"width": 2, "height": 1, "channels": 1, "data": {"$blob": 0, "dtype": "uint8"}}, [b"\x00\xff"])
        self.assertIsInstance(m, RawImage)
        self.assertEqual((m.width, m.height, m.channels), (2, 1, 1))

    def test_functions_become_callbacks(self):
        callbacks = []
        f = lambda *a: None
        self.assertEqual(encode({"streamer": f}, [], callbacks), {"streamer": {"$callback": 0}})
        self.assertIs(callbacks[0], f)

    def test_unknown_values_are_refused(self):
        with self.assertRaises(KilnError):
            encode(object(), [], [])

    def test_tags_in_a_models_own_text_stay_plain(self):
        """A tool call's arguments can hold the codec's keys; only Kiln's exact tags are read as tags."""
        from kiln_webgpu._codec import decode
        for forged in ({"$blob": 3, "dtype": "uint8"}, {"$blob": "x", "dtype": "uint8"}, {"$blob": 0, "dtype": "nope"}, {"$float": 1}, {"$float": "1e999"}, {"$bigint": "9" * 5000}, {"$bigint": "12", "x": 1}):
            self.assertEqual(decode(forged, [b"abc"]), forged)
        self.assertEqual(decode({"$bigint": "-12"}, []), -12)
        self.assertEqual(decode({"$float": "-Infinity"}, []), float("-inf"))

    def test_gpu_off_on_old_macos(self):
        from kiln_webgpu._bridge import gpu_blocked

        self.assertIn("macOS 15", gpu_blocked("Darwin", "22.6.0"))
        self.assertIsNone(gpu_blocked("Darwin", "24.1.0"))
        self.assertIsNone(gpu_blocked("Linux", "6.8.0"))
        self.assertIsNone(gpu_blocked("Windows", "10"))

    def test_png(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.png")
            RawImage(2, 2, 3, bytes(range(12))).save(path)
            with open(path, "rb") as f:
                data = f.read()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(data[-8:-4], b"IEND")

    def test_wav(self):
        import wave

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.wav")
            kiln_webgpu.write_wav(path, [0.0, 1.0, -1.0, 2.0], 16000)
            w = wave.open(path, "rb")
            try:
                self.assertEqual((w.getframerate(), w.getnframes(), w.getnchannels()), (16000, 4, 1))
                self.assertEqual(array.array("h", w.readframes(4)).tolist() if sys.byteorder == "little" else None, [0, 32767, -32767, 32767])
            finally:
                w.close()

    def test_paths_in_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.wav")
            with open(path, "wb") as f:
                f.write(b"RIFFdata")
            args = Pipeline(1, "automatic-speech-recognition", None)._args((path,), {"language": "en"})
            blobs = []
            self.assertEqual(encode(args, blobs, []), [{"$blob": 0, "dtype": "uint8"}, {"language": "en"}])
            self.assertEqual(blobs, [b"RIFFdata"])
            # A sentence stays a sentence for a text task, and so does an audio path there.
            self.assertEqual(Pipeline(1, "text-classification", None)._args((path,), {}), [path])
            part = _media([{"role": "user", "content": [{"type": "audio", "audio": path}, {"type": "text", "text": path}]}])[0]["content"]
            self.assertTrue(isinstance(part[0]["audio"], os.PathLike))
            self.assertEqual(part[1]["text"], path)


@unittest.skipUnless(have_node(), "needs Node %d.%d+ and Kiln's build (npm run build:package)" % MIN_NODE)
class ThroughNode(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        kiln_webgpu.shutdown()

    def test_runtime(self):
        r = kiln_webgpu.runtime()
        self.assertIn("text-generation", r["tasks"])

    def test_values_round_trip(self):
        got = BRIDGE.request("echo", {"args": [{"n": 1, "big": 2**60, "inf": float("inf")}, array.array("f", [0.5, 1.5]), b"\x07", array.array("q", [2**40])]})
        self.assertEqual(got[0], {"n": 1, "big": 2**60, "inf": float("inf")})
        self.assertEqual((got[1].dtype, got[1].tolist()), ("float32", [0.5, 1.5]))
        self.assertEqual((got[2].dtype, got[2].data), ("uint8", b"\x07"))
        self.assertEqual((got[3].dtype, got[3].tolist()), ("int64", [2**40]))

    def test_large_array(self):
        a = array.array("f", range(300000))
        got = BRIDGE.request("echo", {"args": [a]})
        self.assertEqual(got[0].data, a.tobytes() if sys.byteorder == "little" else got[0].data)

    def test_callbacks(self):
        seen = []
        BRIDGE.request("echo", {"args": ["hello", lambda *a: seen.append(a)]})
        self.assertEqual(seen, [("hello",)])

    def test_exported_function(self):
        self.assertAlmostEqual(kiln_webgpu.invoke("cos_sim", array.array("f", [1, 0]), array.array("f", [1, 0])), 1.0)
        self.assertAlmostEqual(kiln_webgpu.invoke("cos_sim", [1, 0], [0, 1]), 0.0)

    def test_run_tools(self):
        """The loop against a scripted model: Kiln's own parser reads the calls."""
        call = lambda name, args: '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (name, args)
        replies = [call("get_weather", '{"city": "Paris"}') + call("boom", "{}") + call("nope", "{}"), "It is 18 degrees."]
        asked = []

        class Scripted:
            def stream(self, messages, **options):
                asked.append((list(messages), options))
                text = replies[len(asked) - 1]
                yield text[:5]
                yield text[5:]

        def boom():
            raise ValueError("no network")

        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}, "execute": lambda city: {"city": city, "celsius": 18}},
            {"type": "function", "function": {"name": "boom"}, "execute": boom},
        ]
        seen = []
        r = kiln_webgpu.run_tools(Scripted(), [{"role": "user", "content": "Weather in Paris?"}], tools, on_text=seen.append, toolChoice="required", max_new_tokens=64)
        self.assertEqual(r["content"], "It is 18 degrees.")
        self.assertFalse(r["unfinished"])
        self.assertEqual([m["role"] for m in r["messages"]], ["user", "assistant", "tool", "tool", "tool", "assistant"])
        self.assertEqual(r["messages"][2]["content"], '{"city": "Paris", "celsius": 18}')
        self.assertEqual([s.get("error") for s in r["steps"]], [None, "no network", "no tool named 'nope'"])
        self.assertEqual("".join(seen), "".join(replies))
        # The model sees the specs without the functions; toolChoice holds for the first reply only.
        self.assertEqual(asked[0][1], {"tools": [{k: v for k, v in t.items() if k != "execute"} for t in tools], "toolChoice": "required", "max_new_tokens": 64})
        self.assertNotIn("toolChoice", asked[1][1])
        with self.assertRaisesRegex(KilnError, "at least one tool"):
            kiln_webgpu.run_tools(Scripted(), [], [])

    def test_run_tools_results_the_model_can_read(self):
        call = lambda name: '<tool_call>\n{"name": "%s", "arguments": {}}\n</tool_call>' % name
        replies = [call("city") + call("when"), "ok"]

        class Scripted:
            n = 0

            def stream(self, messages, **options):
                Scripted.n += 1
                self.options = options
                yield replies[Scripted.n - 1]

        import datetime
        tools = [
            {"type": "function", "function": {"name": "city"}, "execute": lambda: {"city": "Niğde"}},
            {"type": "function", "function": {"name": "when"}, "execute": lambda: datetime.datetime(2026, 1, 1)},
        ]
        chat = Scripted()
        r = kiln_webgpu.run_tools(chat, [{"role": "user", "content": "?"}], tools, tool_choice="required")
        self.assertEqual(r["messages"][2]["content"], '{"city": "Niğde"}')
        self.assertIn("not JSON serializable", r["steps"][1]["error"])
        self.assertIn("not JSON serializable", r["messages"][3]["content"])
        self.assertNotIn("tool_choice", chat.options)
        with self.assertRaises(kiln_webgpu.KilnError):
            kiln_webgpu.run_tools(chat, [], tools + [tools[0]])

    def test_built_in_skills(self):
        tools = kiln_webgpu.skills(max_chars=500)
        self.assertEqual([t["function"]["name"] for t in tools], ["calculate", "get_date_time", "search_wikipedia", "read_web_page"])
        by = {t["function"]["name"]: t["execute"] for t in tools}
        self.assertEqual(by["calculate"](expression="1234 * 5678 + 99")["result"], 7006751)
        self.assertIn("weekday", by["get_date_time"](place="Tokyo"))
        with self.assertRaisesRegex(KilnError, "time zone"):
            by["get_date_time"](place="Nowhereville")
        # The reader refuses this machine before any request is made.
        with self.assertRaises(KilnError):
            by["read_web_page"](url="http://localhost:1/")
        self.assertEqual(kiln_webgpu.calculator_skill()["function"]["name"], "calculate")
        self.assertEqual(kiln_webgpu.wikipedia_skill(language="tr")["function"]["name"], "search_wikipedia")
        with self.assertRaisesRegex(KilnError, "language"):
            kiln_webgpu.wikipedia_skill(language="not a code")
        with self.assertRaisesRegex(KilnError, "unknown skill option"):
            kiln_webgpu.skills(fetch=print)
        self.assertIn("tool", kiln_webgpu.skill_instructions().lower())

    def test_run_tools_with_a_built_in_skill(self):
        replies = ['<tool_call>\n{"name": "calculate", "arguments": {"expression": "6 * 7"}}\n</tool_call>', "It is 42."]

        class Scripted:
            n = 0

            def stream(self, messages, **options):
                Scripted.n += 1
                yield replies[Scripted.n - 1]

        r = kiln_webgpu.run_tools(Scripted(), [{"role": "user", "content": "6 times 7?"}], kiln_webgpu.skills())
        self.assertEqual(r["steps"][0]["result"]["result"], 42)
        self.assertEqual(r["messages"][2]["content"], '{"expression": "6 * 7", "result": 42}')
        self.assertEqual(r["content"], "It is 42.")

    def test_errors_come_back(self):
        with self.assertRaisesRegex(KilnError, "no function"):
            kiln_webgpu.invoke("no_such_function")
        # The process is still there after an error.
        self.assertIn("node", kiln_webgpu.runtime())

    def test_an_abandoned_stream_leaves_the_bridge_usable(self):
        events = BRIDGE.events("echo", {"args": ["x", lambda *a: None]})
        next(events)
        events.close()
        self.assertIn("node", kiln_webgpu.runtime())

    def test_unknown_task(self):
        with self.assertRaises(KilnError):
            kiln_webgpu.pipeline("no-such-task")


if __name__ == "__main__":
    unittest.main()
