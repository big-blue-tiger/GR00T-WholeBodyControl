"""Encode each committed rollout frame once for the MP4 and live preview."""

from fractions import Fraction

import av
import msgpack
import numpy as np
import zmq

from .render_pipeline import atomic_json, PixelStats


class EpisodeVideo:
    def __init__(self, directory, episode_id, preview):
        self.directory = directory
        self.episode_id = episode_id
        self.preview = preview
        self.stats = PixelStats()
        directory.mkdir(parents=True)
        self.partial = directory / "ego_view.partial.mp4"
        self.container = av.open(str(self.partial), "w")
        self.stream = self.container.add_stream("libx264", rate=50)
        self.stream.width, self.stream.height, self.stream.pix_fmt = 640, 480, "yuv420p"
        self.stream.codec_context.time_base = Fraction(1, 50)
        self.stream.codec_context.thread_count = 2
        self.stream.options = {
            "crf": "18",
            "preset": "fast",
            "tune": "zerolatency",
            # Each IDR includes headers so a viewer can join in the middle of an episode.
            "x264-params": "keyint=20:min-keyint=20:scenecut=0:bframes=0:repeat-headers=1:annexb=1",
        }

    def write(self, step, rgb):
        self.stats.add(rgb)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = step, Fraction(1, 50)
        for packet in self.stream.encode(frame):
            self.publish_and_mux(packet)

    def publish_and_mux(self, packet):
        if self.preview is not None:
            # Take the access unit before the MP4 muxer changes its packet timestamps.
            self.preview.send(msgpack.packb(dict(
                episode_id=self.episode_id, step=packet.pts, keyframe=packet.is_keyframe, h264=bytes(packet),
            ), use_bin_type=True), flags=zmq.NOBLOCK)
        self.container.mux(packet)

    def finish(self, timestamps):
        for packet in self.stream.encode():
            self.publish_and_mux(packet)
        self.close()
        count = len(timestamps)
        with av.open(str(self.partial)) as container:
            stream = container.streams.video[0]
            if self.stats.frames != count or stream.frames != count or stream.average_rate != 50:
                raise ValueError(f"Video does not match {count} control steps: {self.partial}")
        self.partial.replace(self.directory / "ego_view.mp4")
        np.savez(self.directory / "frames.npz", frame_index=np.arange(count), timestamp=timestamps)
        atomic_json(self.directory / "render.json", dict(
            frames=count, fps=50, width=640, height=480, video_stats=self.stats.result(),
        ))

    def close(self):
        self.container.close()
