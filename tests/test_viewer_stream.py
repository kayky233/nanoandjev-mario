"""A late viewer must receive a decodable image after emulation has stopped."""

import http.client
import io
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image

import watch_local as viewer


class ViewerStreamTests(unittest.TestCase):
    def test_model_only_cannot_use_routes_guards_or_fallback(self):
        for source, error in (("model", None), ("rules_fallback", "ValueError")):
            with self.subTest(source=source), \
                 patch.object(viewer.P, "MAX_FRAMES", 7), \
                 patch.object(viewer.P, "ask_local", return_value=("run right", None, 0, 0.01, source)) as ask, \
                 patch.object(viewer.P, "hazard_guard", side_effect=AssertionError("guard forbidden")), \
                 patch.object(viewer.P, "unstick", side_effect=AssertionError("unstick forbidden")), \
                 patch.object(viewer.P, "policy", side_effect=AssertionError("rules forbidden")), \
                 patch.object(viewer, "publish") as publish:
                result = viewer.play_episode("1-1", 1_000_000, "chat", 1, sync=True, model_only=True)
                self.assertFalse(result["verified_route_used"])
                self.assertEqual(result["verified_route_steps"], [])
                self.assertEqual(result["mentor_trace"], [])
                self.assertEqual(result["error"], error)
                self.assertTrue(ask.call_args.kwargs["strict"])
                if error:
                    self.assertEqual(result["frames"], 0)
                    self.assertEqual(result["model_decision_sources"], {"error": 1})
                    self.assertIsNone(publish.call_args.args[1]["decision"]["choice"])
                else:
                    self.assertEqual(publish.call_args.args[1]["decision"]["choice"], "run right")
        with patch.object(viewer, "JoypadSpace") as env:
            for kwargs in ({"verified_route": True, "sync": True}, {"sync": False}, {"model": "replay", "sync": True}):
                with self.assertRaisesRegex(ValueError, "requires a synchronous model"):
                    viewer.play_episode("1-2", 60, "chat", 1, model_only=True, **kwargs)
            env.assert_not_called()

    def test_completed_episode_sends_next_boundary_without_new_game_frames(self):
        image = io.BytesIO()
        Image.new("RGB", (256, 240), "blue").save(image, format="JPEG")
        with viewer.LOCK:
            previous = viewer.LATEST["local"]
            viewer.LATEST["local"] = {"seq": 1, "jpeg": image.getvalue(), "meta": {"flag": True}}
        server = ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            client.request("GET", "/stream.mjpg")
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            body = b""
            boundary = b"--" + viewer.BOUNDARY
            while body.count(boundary) < 2:
                chunk = response.read1(4096)
                self.assertTrue(chunk)
                body += chunk
            part = body.split(boundary)[1]
            jpeg = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
            with Image.open(io.BytesIO(jpeg)) as decoded:
                self.assertEqual(decoded.size, (256, 240))
                decoded.load()
            self.assertEqual(viewer.LATEST["local"]["seq"], 1)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
            with viewer.LOCK:
                viewer.LATEST["local"] = previous


if __name__ == "__main__":
    unittest.main()
