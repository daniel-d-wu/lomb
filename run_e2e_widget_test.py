"""
run_e2e_widget_test.py -- genuine end-to-end test of the active-enrollment
slice: a REAL browser (Chromium via Playwright, fake mic) opens
enroll_widget.html served by a REAL running enroll_api.py FastAPI app,
records >=8s of fake audio through getUserMedia/MediaRecorder, uploads it
to /enroll, and the assertions check the actual HTTP/JSON response and the
actual SQLite-backed /status afterward.

This deliberately does NOT unit-test in isolation -- test_voice_enrollment.py
already covers the resolution/threshold logic with a FakeEmbedder, and
audio_decode.py has its own direct ffmpeg round-trip test. What NEITHER of
those touches is the seam this script exists to prove: does a real browser
MediaRecorder blob, sent as a real multipart upload, over a real running
uvicorn server, actually decode and enroll end-to-end.

Uses a trivial embedder (not speechbrain -- torch/speechbrain aren't
installed in this sandbox, and enroll_active()'s logic under test here
doesn't depend on embedding quality, just that SOME fixed-length vector
comes back for a real decoded waveform).

Run: python run_e2e_widget_test.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_server_module(db_path: str) -> Path:
    """A tiny module (not enroll_api.py itself) that wires create_app() with
    a fake embedder + real SQLite repo, so uvicorn can import it by
    module:attr. Keeps the fake embedder out of production code entirely."""
    server_src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(HERE)!r})

        import numpy as np
        from enroll_api import create_app
        from voice_enrollment import SpeakerEmbedder, SQLiteVoiceprintRepository


        class FixedLengthFakeEmbedder(SpeakerEmbedder):
            \"\"\"Deterministic stand-in for EcapaSpeakerEmbedder -- returns a
            192-dim vector derived from simple waveform statistics. Good
            enough to exercise enroll_active()'s storage path; this test
            doesn't check embedding quality, just that a real decoded
            waveform makes it all the way to a stored voiceprint.\"\"\"

            def embed(self, waveform: np.ndarray) -> np.ndarray:
                rng = np.random.default_rng(abs(hash(waveform.tobytes())) % (2**32))
                vec = rng.normal(size=192).astype(np.float32)
                vec[0] += float(waveform.mean())
                vec[1] += float(waveform.std())
                return vec


        app = create_app(FixedLengthFakeEmbedder(), SQLiteVoiceprintRepository({db_path!r}))
    """)
    path = HERE / "_e2e_server_module.py"
    path.write_text(server_src)
    return path


def wait_for_server(base_url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(base_url + "/status", timeout=1)
            if r.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.2)
    raise RuntimeError(f"server never came up at {base_url}: {last_err}")


def main() -> int:
    db_path = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name
    write_server_module(db_path)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"

    print(f"[1/8] starting uvicorn on {base_url} ...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "_e2e_server_module:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(HERE),
    )

    try:
        wait_for_server(base_url)
        print("      server is up.")

        print("[2/8] confirming /status shows NOT enrolled yet ...")
        status_before = requests.get(base_url + "/status").json()
        assert status_before == {"enrolled": False, "sample_count": 0}, status_before
        print(f"      {status_before}")

        print("[3/8] launching real Chromium with fake mic device via Playwright ...")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",  # auto-grant mic permission prompt
                ],
            )
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()

            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            print(f"[4/8] navigating to {base_url}/ and starting a recording ...")
            page.goto(base_url + "/")
            page.wait_for_selector("#record-btn")
            assert "No voiceprint on file yet" in page.text_content("#status-line")

            page.click("#record-btn")
            page.wait_for_selector("#record-btn.recording", timeout=3000)
            print("      recording started, waiting 9s (min enrollment is 8s) ...")
            page.wait_for_timeout(9000)

            print("[5/8] stopping recording and waiting for /enroll upload to complete ...")
            page.click("#record-btn")
            page.wait_for_selector("#message.success", timeout=20000)
            message_text = page.text_content("#message")
            print(f"      widget message: {message_text!r}")
            assert "Voice enrolled from a" in message_text

            status_line_after = page.text_content("#status-line")
            print(f"      status line after enroll: {status_line_after!r}")
            assert "already on file" in status_line_after

            if console_errors:
                print(f"      (non-fatal) console errors seen: {console_errors}")

            browser.close()

        print("[6/8] confirming /status now shows enrolled via direct HTTP call ...")
        status_after = requests.get(base_url + "/status").json()
        print(f"      {status_after}")
        assert status_after["enrolled"] is True
        assert status_after["sample_count"] == 4  # default initial_weight in enroll_active()

        print("[7/8] re-enrolling WITHOUT force through the widget -- must be refused ...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
            )
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()
            page.goto(base_url + "/")
            page.wait_for_selector("#record-btn")
            assert "already on file" in page.text_content("#status-line")
            # force checkbox must be visible (and left UNCHECKED) since a voiceprint exists
            assert page.is_visible("#force-row")
            assert not page.is_checked("#force-checkbox")

            page.click("#record-btn")
            page.wait_for_timeout(9000)
            page.click("#record-btn")
            page.wait_for_selector("#message.error", timeout=20000)
            refused_message = page.text_content("#message")
            print(f"      widget message: {refused_message!r}")
            assert "Error:" in refused_message
            browser.close()

        unchanged = requests.get(base_url + "/status").json()
        assert unchanged == {"enrolled": True, "sample_count": 4}
        print(f"      confirmed unchanged: {unchanged}")

        print("[8/8] re-enrolling WITH force checked through the widget -- must replace cleanly ...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
            )
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()
            page.goto(base_url + "/")
            page.wait_for_selector("#record-btn")
            page.check("#force-checkbox")

            page.click("#record-btn")
            page.wait_for_timeout(9000)
            page.click("#record-btn")
            page.wait_for_selector("#message.success", timeout=20000)
            forced_message = page.text_content("#message")
            print(f"      widget message: {forced_message!r}")
            assert "Voice enrolled from a" in forced_message
            browser.close()

        replaced = requests.get(base_url + "/status").json()
        print(f"      {replaced}")
        assert replaced == {"enrolled": True, "sample_count": 4}  # fresh initial_weight, not additive

        print(
            "\nPASS: full active-enrollment slice verified end-to-end through a real browser -- "
            "first enroll, refused re-enroll without force, and force=True reset all confirmed via /status."
        )
        return 0

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        Path(HERE / "_e2e_server_module.py").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
