# ros_image_reader.py
# Read images from the ROS topics defined in your communication protocol.
# Subscribes to:
#   - /jhu_daVinci/left/image_raw   (primary endoscope - 1080p)
#   - /jhu_daVinci/right/image_raw  (optional endoscope)
#   - /PSM1/endoscope_img           (optional wrist camera)
#   - /PSM2/endoscope_img           (optional wrist camera)
#   - /yolo_segmentation_masks      (optional segmentation mask image)
#
# Exposes a simple API to get the latest frames as NumPy arrays and to save them to disk.

import os
import time
import threading
import tempfile
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class _TopicFrameBuffer:
    """Thread-safe container for a single topic's latest image frame."""
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._bgr: Optional[np.ndarray] = None
        self._stamp: Optional[rospy.Time] = None

    def update(self, bgr: np.ndarray, stamp: rospy.Time):
        with self._lock:
            self._bgr = bgr
            self._stamp = stamp

    def has_frame(self) -> bool:
        with self._lock:
            return self._bgr is not None

    def get(self) -> Optional[Tuple[np.ndarray, rospy.Time]]:
        with self._lock:
            if self._bgr is None:
                return None
            return self._bgr.copy(), self._stamp


class ROSImageReader:
    """
    Subscribe to image topics and provide access to the latest frames.
    Only reads images; does not publish or run inference.

    Default topics (all optional except left camera):
      - left:  /jhu_daVinci/left/image_raw
      - right: /jhu_daVinci/right/image_raw
      - psm1:  /PSM1/endoscope_img
      - psm2:  /PSM2/endoscope_img
      - seg:   /yolo_segmentation_masks (grayscale or multi-class mask as Image)

    Usage:
        reader = ROSImageReader()
        reader.start()  # init node + subscribers
        ok = reader.wait_for_any(["left"], timeout_sec=5.0)
        frame, t = reader.get("left")
        path = reader.save("left", "/tmp")
    """

    DEFAULT_TOPICS = {
        "left":  "/jhu_daVinci/left/image_raw",
        "right": "/jhu_daVinci/right/image_raw",
        "psm1":  "/PSM1/endoscope_img",
        "psm2":  "/PSM2/endoscope_img",
        "seg":   "/yolo_segmentation_masks",
    }

    def __init__(
        self,
        node_name: str = "image_reader",
        topics: Optional[Dict[str, str]] = None,
        subscribe_right: bool = True,
        subscribe_psm1: bool = True,
        subscribe_psm2: bool = True,
        subscribe_seg: bool = False,
    ):
        """
        Args:
            node_name: ROS node name.
            topics: Optional override for topic names (dict keys: left/right/psm1/psm2/seg).
            subscribe_*: Enable/disable optional topics.
        """
        self.node_name = node_name
        self.bridge = CvBridge()
        t = dict(self.DEFAULT_TOPICS)
        if topics:
            t.update(topics)
        self.topic_map = t

        # Which topics to subscribe to
        self.enabled_keys = ["left"]
        if subscribe_right: self.enabled_keys.append("right")
        if subscribe_psm1: self.enabled_keys.append("psm1")
        if subscribe_psm2: self.enabled_keys.append("psm2")
        if subscribe_seg:  self.enabled_keys.append("seg")

        # Per-topic buffers
        self.buffers: Dict[str, _TopicFrameBuffer] = {
            key: _TopicFrameBuffer(name=key) for key in self.enabled_keys
        }

        self._subs = []
        self._started = False

    def start(self, anonymous: bool = True) -> None:
        """Initialize ROS node (if needed) and create subscribers for enabled topics."""
        if not rospy.core.is_initialized():
            rospy.init_node(self.node_name, anonymous=anonymous)

        if self._started:
            return

        for key in self.enabled_keys:
            topic = self.topic_map[key]
            sub = rospy.Subscriber(topic, Image, self._make_cb(key), queue_size=1)
            self._subs.append(sub)
        self._started = True

    def _make_cb(self, key: str):
        """Build a callback that converts Image to BGR np.ndarray (or mask) and stores it."""
        def _cb(msg: Image):
            try:
                # For raw camera topics, keep as BGR (OpenCV native).
                if key in ("left", "right", "psm1", "psm2"):
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    if cv_img.ndim == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                else:
                    # For segmentation, keep as single-channel if possible to preserve labels.
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    if cv_img.ndim == 3 and cv_img.shape[2] == 3:
                        # If the seg is accidentally RGB, convert to BGR for consistency.
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    elif cv_img.ndim == 2:
                        # Keep 1-channel; store as 3-channel for a consistent return type if needed.
                        # Here we keep original (grayscale) to not lose label IDs.
                        pass
                stamp = msg.header.stamp if msg.header else rospy.Time.now()
            except Exception as e:
                rospy.logerr(f"[ROSImageReader:{key}] cv_bridge conversion failed: {repr(e)}")
                return

            # Ensure numpy array and types are sane
            cv_img = np.asarray(cv_img)
            self.buffers[key].update(cv_img, stamp)
        return _cb

    # ---------------- Public API ----------------

    def has(self, key: str) -> bool:
        """Return True if the given topic key has at least one frame."""
        self._ensure_key(key)
        return self.buffers[key].has_frame()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, rospy.Time]]:
        """
        Get the latest frame for a topic key.

        Returns:
            (image, stamp) or None if not available.
            - For 'left'/'right'/'psm1'/'psm2': image is BGR (H, W, 3), uint8.
            - For 'seg': image may be 1-channel label map or BGR depending on publisher.
        """
        self._ensure_key(key)
        return self.buffers[key].get()

    def wait_for(self, key: str, timeout_sec: float = 5.0) -> Optional[Tuple[np.ndarray, rospy.Time]]:
        """Block until a frame arrives for one topic or timeout."""
        self._ensure_key(key)
        t0 = time.time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            item = self.get(key)
            if item is not None:
                return item
            if (time.time() - t0) >= timeout_sec:
                return None
            rate.sleep()

    def wait_for_any(self, keys: Optional[list] = None, timeout_sec: float = 5.0) -> Optional[Tuple[str, np.ndarray, rospy.Time]]:
        """
        Block until any of the given topics (or all enabled if None) has a frame.

        Returns:
            (key, image, stamp) or None on timeout.
        """
        keys = keys or self.enabled_keys
        t0 = time.time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            for k in keys:
                got = self.get(k)
                if got is not None:
                    img, ts = got
                    return k, img, ts
            if (time.time() - t0) >= timeout_sec:
                return None
            rate.sleep()

    def save(self, key: str, save_dir: Optional[str] = None, prefix: Optional[str] = None) -> Optional[str]:
        """
        Save the latest frame of a topic to disk as .jpg or .png.

        Returns:
            output file path, or None if no frame available.
        """
        self._ensure_key(key)
        item = self.get(key)
        if item is None:
            return None
        img, _ = item

        if save_dir is None:
            save_dir = tempfile.gettempdir()
        os.makedirs(save_dir, exist_ok=True)

        if prefix is None:
            prefix = f"{key}_"

        ts_ms = int(time.time() * 1000)
        # If seg is single-channel integer labels, prefer PNG to preserve values.
        ext = ".png" if (key == "seg" and (img.ndim == 2 or img.shape[2] == 1)) else ".jpg"
        out_path = os.path.join(save_dir, f"{prefix}{ts_ms}{ext}")

        # For single-channel seg map, ensure we write properly (PNG keeps indices).
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ok = cv2.imwrite(out_path, img)
        else:
            # BGR or 3-channel seg
            ok = cv2.imwrite(out_path, img)

        if not ok:
            raise IOError(f"Failed to write image to {out_path}")
        return out_path

    # ---------------- Internals ----------------

    def _ensure_key(self, key: str):
        if key not in self.buffers:
            raise KeyError(
                f"Topic key '{key}' not enabled. Enabled keys: {self.enabled_keys}. "
                f"Pass subscribe_* flags in the constructor or provide a topics dict."
            )


# ----------------------------- CLI demo -----------------------------

if __name__ == "__main__":
    """
    Minimal CLI:
      1) Starts subscribers.
      2) Waits for a frame on left camera (or any enabled topic with --any).
      3) Saves the frame to disk and prints the file path.

    Examples:
      # Left camera only:
      python ros_image_reader.py

      # Enable right and both wrist cameras:
      python ros_image_reader.py --right --psm1 --psm2

      # Wait for any topic among left/right and save:
      python ros_image_reader.py --right --any --timeout 3 --out /tmp
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROS image reader (subscribe-only)")
    parser.add_argument("--node_name", type=str, default="image_reader", help="ROS node name")
    parser.add_argument("--right", action="store_true", help="Subscribe to /jhu_daVinci/right/image_raw")
    parser.add_argument("--psm1", action="store_true", help="Subscribe to /PSM1/endoscope_img")
    parser.add_argument("--psm2", action="store_true", help="Subscribe to /PSM2/endoscope_img")
    parser.add_argument("--seg", action="store_true", help="Subscribe to /yolo_segmentation_masks")
    parser.add_argument("--any", action="store_true", help="Wait for any enabled topic instead of 'left'")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for a frame")
    parser.add_argument("--out", type=str, default=None, help="Directory to save the captured frame")
    args = parser.parse_args()

    reader = ROSImageReader(
        node_name=args.node_name,
        subscribe_right=args.right,
        subscribe_psm1=args.psm1,
        subscribe_psm2=args.psm2,
        subscribe_seg=args.seg,
    )
    reader.start()

    if args.any:
        got = reader.wait_for_any(timeout_sec=args.timeout)
        if got is None:
            print("Timeout waiting for any frame.")
        else:
            k, img, ts = got
            path = reader.save(k, save_dir=args.out) or "<no frame>"
            print(f"[{k}] stamp={ts.to_sec() if ts else 'n/a'}, saved to: {path}")
    else:
        got = reader.wait_for("left", timeout_sec=args.timeout)
        if got is None:
            print("Timeout waiting for left frame.")
        else:
            img, ts = got
            path = reader.save("left", save_dir=args.out) or "<no frame>"
            print(f"[left] stamp={ts.to_sec() if ts else 'n/a'}, saved to: {path}")