import argparse
import os
import shutil
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union

from math import exp

# IMPORTANT: This function must return a probability in [0,1].
# Numeric strings and bool are tolerated, but the intended contract is a float probability.
from model_prob import check_task_complete

# ROS utilities
from ros_image_reader import ROSImageReader
from ros_command_publisher import InstructorCommandPublisher


Number = Union[int, float]


class ClosedLoopController:
    """
    Closed-loop high-level controller with ROS (no human verification).

    New behavior:
      • Use a per-task logistic threshold schedule:
          T(n) = T_min + (T0 - T_min) / (1 + exp(a * (n - K - b)))
        where n is the consecutive-failure count for the current task.
        T0 and K are configurable per task (1..5). Task 6 uses a fixed threshold 0.5.

      • The API function 'check_task_complete(image_path, task_id, api_url)'
        is expected to return a single probability in [0,1].
        (We still accept bool or numeric strings for robustness.)

      • Cycle 1 -> 2 -> 3 -> 4 -> 5 -> (wrap) -> 1...
        Only after Task 1 is complete do we check Task 6 (finish gate, T=0.5):
          - If finished: optionally send '6_finished' and EXIT.
          - Else: continue to Task 2.

      • Only './dvrk_left.jpg' is kept on disk (overwritten each tick).
      • Each TURN has one header with timestamp; lines inside the TURN have no per-line timestamps.
    """

    # -------------------- init --------------------

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # --- phases
        self.current_task_id = 1
        self.first_phase_id = 1
        self.last_phase_id = 5
        self.final_done_id = 6

        self.task_id_to_cmd = {
            1: "1_retract_home_in",
            2: "2_resect_home_in",
            3: "3_resect",
            4: "4_resect_home_out",
            5: "5_retract_home_out",
            6: "6_finished",
        }

        # --- API / ROS config
        self.api_url = cfg.get("vlm_api", None)
        self.send_final_on_exit = bool(cfg.get("send_final_on_exit", True))

        subscribe_right = bool(cfg.get("subscribe_right", False))
        subscribe_psm1  = bool(cfg.get("subscribe_psm1", False))
        subscribe_psm2  = bool(cfg.get("subscribe_psm2", False))
        subscribe_seg   = bool(cfg.get("subscribe_seg", False))

        self.frame_timeout = float(cfg.get("frame_timeout_sec", 5.0))
        self.save_dir = cfg.get("ros_frame_save_dir", "./")  # always use ./

        # --- per-task poll intervals
        self.default_poll_interval = float(cfg.get("poll_interval_sec", 20.0))
        raw_overrides = cfg.get("per_task_poll_intervals", {}) or {}
        self.per_task_poll_intervals: Dict[int, float] = {}
        for k, v in raw_overrides.items():
            try:
                self.per_task_poll_intervals[int(k)] = float(v)
            except Exception:
                pass

        # --- threshold schedule config
        thr_cfg = cfg.get("threshold_cfg", {}) or {}
        self.T_min: float = float(thr_cfg.get("T_min", 0.30))
        self.a: float     = float(thr_cfg.get("a", 1.2))
        self.b: float     = float(thr_cfg.get("b", 0.0))
        raw_T0 = (thr_cfg.get("per_task_T0", {}) or {})
        raw_K  = (thr_cfg.get("per_task_K", {}) or {})
        self.per_task_T0: Dict[int, float] = {}
        self.per_task_K:  Dict[int, int]   = {}
        for k, v in raw_T0.items():
            try:
                self.per_task_T0[int(k)] = float(v)
            except Exception:
                pass
        for k, v in raw_K.items():
            try:
                self.per_task_K[int(k)] = int(v)
            except Exception:
                pass
        self.default_T0: float = float(thr_cfg.get("default_T0", 0.80))
        self.default_K:  int   = int(thr_cfg.get("default_K", 0))

        # Finish-gate fixed threshold
        self.finish_gate_threshold: float = float(thr_cfg.get("finish_gate_T", 1.01))

        # Consecutive failure streak per task (1..5)
        self.fail_streak: Dict[int, int] = {i: 0 for i in range(1, self.last_phase_id + 1)}

        # --- logging
        logs_dir = cfg.get("logs_dir", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(logs_dir, f"closed_loop_{ts}.log")
        self._setup_logger(self.log_path)
        self.turn = 0

        # --- ROS
        self.reader = ROSImageReader(
            node_name=cfg.get("ros_node_name", "image_reader_closed_loop"),
            subscribe_right=subscribe_right,
            subscribe_psm1=subscribe_psm1,
            subscribe_psm2=subscribe_psm2,
            subscribe_seg=subscribe_seg,
        )
        self.reader.start()

        self.cmd_pub = InstructorCommandPublisher(
            node_name=cfg.get("ros_cmd_node_name", "instructor_publisher"),
            instruction_topic="/instructor_prediction",
            queue_size=10
        )

        # Init logs
        self.log(f"[INIT] Logging to: {self.log_path}")
        self.log(f"[INIT] Default poll interval: {self.default_poll_interval:.1f}s; overrides: {self.per_task_poll_intervals}")
        self.log(f"[INIT] Thresholds: T_min={self.T_min:.3f}, a={self.a:.3f}, b={self.b:.3f}, "
                 f"default_T0={self.default_T0:.3f}, default_K={self.default_K}, "
                 f"per_task_T0={self.per_task_T0}, per_task_K={self.per_task_K}, "
                 f"finish_gate_T={self.finish_gate_threshold:.3f}")

    # -------------------- logging helpers --------------------

    def _setup_logger(self, logfile: str) -> None:
        self.logger = logging.getLogger("closed_loop")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logfile)
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(fmt="%(message)s")  # no per-line timestamps
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.handlers.clear()
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def log_turn_header(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 64
        self.log(f"{now} — TURN {self.turn}")
        self.log(sep)

    # -------------------- timing helpers --------------------

    def get_poll_interval_for_task(self, task_id: int) -> float:
        return float(self.per_task_poll_intervals.get(task_id, self.default_poll_interval))

    def sleep_with_countdown(self, seconds: float) -> None:
        secs = int(max(0, round(seconds)))
        self.log(f"[WAIT] Sleeping {secs}s before next tick...")
        for remaining in range(secs, 0, -1):
            print(f"\r[WAIT] {remaining:2d}s remaining...", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 30 + "\r", end="", flush=True)
        self.log("[WAIT] Done.")

    # -------------------- threshold schedule --------------------

    def _get_T0_K_for_task(self, task_id: int) -> Tuple[float, int]:
        T0 = self.per_task_T0.get(task_id, self.default_T0)
        K  = self.per_task_K.get(task_id, self.default_K)
        return T0, K

    def threshold_T(self, task_id: int, n_fail: int) -> float:
        """
        T(n) = T_min + (T0 - T_min) / (1 + exp(a * (n - K - b)))
        """
        T0, K = self._get_T0_K_for_task(task_id)
        val = self.T_min + (T0 - self.T_min) / (1.0 + exp(self.a * (n_fail - K - self.b)))
        # Clamp to [T_min, T0]
        if val < self.T_min:
            val = self.T_min
        if val > T0:
            val = T0
        return float(val)

    # -------------------- image & ROS helpers --------------------

    def get_current_obs(self) -> str:
        got = self.reader.wait_for("left", timeout_sec=self.frame_timeout)
        if got is None:
            raise TimeoutError(
                f"No image received from /jhu_daVinci/left/image_raw within {self.frame_timeout} seconds."
            )
        temp_path  = self.reader.save("left", save_dir=self.save_dir)
        fixed_path = os.path.join(self.save_dir or "./", "dvrk_left.jpg")
        try:
            if os.path.abspath(temp_path) != os.path.abspath(fixed_path):
                shutil.copyfile(temp_path, fixed_path)
                try:
                    os.remove(temp_path)
                except Exception as e_rm:
                    self.log(f"[HL] Warning: could not remove temp image '{temp_path}': {repr(e_rm)}")
            # self.log(f"[HL] Saved current frame to: {fixed_path}")
            return fixed_path
        except Exception as e:
            self.log(f"[HL] Failed to write fixed image '{fixed_path}' from '{temp_path}': {repr(e)}")
            raise

    def _task_id_to_command_str(self, task_id: int) -> str:
        cmd = self.task_id_to_cmd.get(int(task_id))
        if not cmd:
            raise ValueError(f"Unknown task_id={task_id}. Valid keys: {sorted(self.task_id_to_cmd.keys())}")
        return cmd

    def send_llp_command(self, task_id: int) -> str:
        cmd = self._task_id_to_command_str(task_id)
        try:
            self.cmd_pub.send_instruction(cmd)
            msg = f"[HL->LL] Sent instruction: '{cmd}'"
            # self.log(msg)
            return msg
        except Exception as e:
            msg = f"[HL->LL] Failed to send instruction '{cmd}': {repr(e)}"
            self.log(msg)
            return msg

    # -------------------- probability handling --------------------

    @staticmethod
    def _parse_probability(x: Any) -> Optional[float]:
        """
        Convert the API return into a probability in [0,1].
        Expected: float in [0,1]. Also accept numeric strings and bool.
        Returns None if it cannot be parsed.
        """
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        if isinstance(x, (int, float)):
            v = float(x)
            return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
        if isinstance(x, str):
            try:
                v = float(x.strip())
                return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
            except Exception:
                return None
        return None

    def check_task_probability(self, img_path: str, task_id: int) -> Optional[float]:
        """
        Call the API and interpret the single return value as probability in [0,1].
        """
        try:
            raw = check_task_complete(
                image_path=img_path,
                task_id=task_id,
                api_url=self.api_url
            )
        except Exception as e:
            self.log(f"[HL] check_task_complete error at task_id={task_id}: {repr(e)}")
            return None

        prob = self._parse_probability(raw)
        if prob is None:
            self.log(f"[API PARSE] task {task_id} -> could not parse probability from value: {raw!r}")
            return None
        return prob

    # -------------------- finish gate --------------------

    def _check_finish_gate_with_image(self, img_path: str) -> bool:
        prob = self.check_task_probability(img_path, self.final_done_id)
        if prob is None:
            return False
        passed = prob >= self.finish_gate_threshold
        self.log(f"[API] task {self.final_done_id} (finish gate) prob={prob:.3f} "
                 f"vs T={self.finish_gate_threshold:.3f} -> complete? {passed}")
        return passed

    # -------------------- main loop --------------------

    def run(self) -> None:
        """
        TURN:
          1) Wait for the configured interval (per current task)
          2) Capture ./dvrk_left.jpg
          3) Get probability for current task & apply T(n) with that task's fail streak
          4) If pass:
               - If just finished task 1: check finish gate (fixed 0.5). If pass -> optional '6_finished' + exit
               - Advance to next task; publish command (or wrap 5->1)
             Else:
               - Increment fail streak for that task; re-send current command
        """
        try:
            # publish initial command
            self.log(self.send_llp_command(self.current_task_id))

            while True:
                self.turn += 1
                self.log_turn_header()

                # 1) per-task wait
                interval = self.get_poll_interval_for_task(self.current_task_id)
                self.sleep_with_countdown(interval)

                # 2) image
                try:
                    img_path = self.get_current_obs()
                except Exception as e:
                    self.log(f"[HL] Failed to capture ROS image at task_id={self.current_task_id}: {repr(e)}")
                    continue

                # 3) probability & decision with T(n)
                n_fail = self.fail_streak.get(self.current_task_id, 0)
                T = self.threshold_T(self.current_task_id, n_fail)
                prob = self.check_task_probability(img_path, self.current_task_id)
                if prob is None:
                    # Treat as error -> keep LLP engaged
                    self.log(self.send_llp_command(self.current_task_id))
                    self.log("[HL] Re-sent current command due to API error/invalid response.")
                    continue
                
                prob = 1-prob

                passed = prob >= T
                self.log(f"[API] task {self.current_task_id} prob={prob:.3f} vs T(n={n_fail})={T:.3f} -> complete? {passed}")

                # 4) act
                if passed:
                    # reset streak for this task
                    self.fail_streak[self.current_task_id] = 0

                    prev_task = self.current_task_id
                    self.current_task_id += 1

                    # If we just completed TASK 1, check finish gate before moving to task 2
                    if prev_task == 1:
                        self.log("[HL] Task 1 completed; checking finish gate (task 6) before moving to task 2...")
                        try:
                            finish_img = self.get_current_obs()
                        except Exception as e:
                            self.log(f"[HL] Failed to capture image for finish gate (after task 1): {repr(e)}")
                        else:
                            if self._check_finish_gate_with_image(finish_img):
                                if self.send_final_on_exit:
                                    self.log(self.send_llp_command(self.final_done_id))  # "6_finished"
                                self.log("[HL] Finish gate confirmed after completing task 1. Exiting.")
                                return

                    # publish next or wrap
                    if self.current_task_id <= self.last_phase_id:
                        self.log(self.send_llp_command(self.current_task_id))
                    else:
                        self.log("[HL] Wrapping 5 -> 1")
                        self.current_task_id = self.first_phase_id
                        self.log(self.send_llp_command(self.current_task_id))

                else:
                    # increment streak and re-send
                    self.fail_streak[self.current_task_id] = n_fail + 1
                    self.log(self.send_llp_command(self.current_task_id))
                    self.log(f"[HL] Phase {self.current_task_id} not complete (fail_streak={self.fail_streak[self.current_task_id]}). Re-sent current command.")

        except KeyboardInterrupt:
            self.log("[HL] Interrupted by user. Shutting down gracefully.")
        except Exception as e:
            self.log(f"[HL] Fatal error: {repr(e)}")
            raise
        finally:
            self.log("[HL] Controller stopped.")


# -------------------- entry point --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Closed-loop ROS controller with logistic threshold schedule and per-task poll intervals"
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=20.0,
        help="Default polling interval in seconds for tasks (unless overridden per task). Default 20s."
    )
    args = parser.parse_args()

    # Example per-task wait overrides
    per_task_overrides = {
        4: 35.0,  # task 4 waits 35s between checks
        5: 15.0,  # task 5 waits 15s between checks
        # others use default (20s)
    }

    # Example per-task threshold parameters (T0 and K); others use defaults
    per_task_T0 = {
        1: 0.7,
        2: 0.7,
        3: 0.87,
        4: 0.7,
        5: 0.70,
    }
    per_task_K = {
        1: 1,
        2: 1,
        3: 2,
        4: 1,
        5: 1,
    }

    cfg = {
        "vlm_api": "http://10.162.34.47:8886",
        "send_final_on_exit": True,

        # ROS image reader / publisher settings
        "ros_node_name": "image_reader_closed_loop",
        "ros_cmd_node_name": "instructor_publisher",

        "subscribe_right": False,
        "subscribe_psm1": False,
        "subscribe_psm2": False,
        "subscribe_seg": False,

        "frame_timeout_sec": 5.0,

        # Always save images to current directory (fixed name)
        "ros_frame_save_dir": "./",

        # Global default poll cadence (overridden by CLI)
        "poll_interval_sec": args.poll_interval_sec,

        # Per-task overrides (keys may be int or str)
        "per_task_poll_intervals": per_task_overrides,

        # Threshold schedule configuration
        "threshold_cfg": {
            "T_min": 0.20,
            "a": 1.2,
            "b": 0.0,
            "default_T0": 0.80,
            "default_K": 0,
            "per_task_T0": per_task_T0,
            "per_task_K": per_task_K,
            # Task 6 uses a fixed threshold:
            "finish_gate_T": 0.99,
        },

        # Logs directory
        "logs_dir": "logs",
    }

    controller = ClosedLoopController(cfg)
    controller.run()
