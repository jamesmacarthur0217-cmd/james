"""Network cameras: read over TCP, and their password never written to the log."""
import logging
import os
import time

from labeler.sources import FrameSource


def test_network_cameras_use_tcp():
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


class _Closed:
    def isOpened(self):
        return False


def test_the_camera_password_is_not_logged(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="labeler.sources")
    monkeypatch.setattr(FrameSource, "_open_live", lambda self: _Closed())   # (a real failed connect takes ~30 s)
    src = FrameSource("rtsp://visionlabeler:s3cretpass@127.0.0.1:9/cam")
    try:
        for _ in range(100):
            if "cannot open" in caplog.text:
                break
            time.sleep(0.1)
    finally:
        src.release()
    assert "cannot open stream rtsp://visionlabeler:•••@127.0.0.1:9/cam" in caplog.text
    assert "s3cretpass" not in caplog.text
