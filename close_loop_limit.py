import argparse
import os
import shutil
import time
from typing import Dict, Any

from model import check_task_complete

# ROS utilities
from ros_image_reader import ROSImageReader
from ros_command_publisher import InstructorCommandPublisher


class ClosedLoopController:
    """Closed-loop high-level controller with ROS (no human verification).

    Cycle: 1->2->3->4->5->(wrap)->1 ...
    After TASK 1 is verified complete, capture a fresh image and check TASK 6 (finish gate).
    If finished, optionally publish "6_finished" and exit; otherwise continue the cycle.

    Added behavior:
    - For tasks 1..5, if verification fails (or the check errors) K times in a row,
      auto-mark the task as success and advance. K is configurable per-task or via a default.

    New features:
    - Start from a configurable task id (default 1).
    - Per-task poll intervals with a global default fallback.
    - A one-second countdown printed before each poll/check.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # Phases 1..5; 6 is the "finish gate"
        self.first_phase_id = 1
        self.last_phase_id = 5
        self.final_done_id = 6

        # Start task id (validated below)
        self.current_task_id = int(cfg.get("start_task_id", 1))
        if self.current_task_id < self.first_phase_id or self.current_task_id > self.last_phase_id:
            raise ValueError(
                f"start_task_id must be in [{self.first_phase_id}..{self.last_phase_id}] "
                f"but got {self.current_task_id}"
            )

        self.task_id_to_cmd = {
            1: "1_retract_home_in",
            2: "2_resect_home_in",
            3: "3_resect",
            4: "4_resect_home_out",
            5: "5_retract_home_out",
            6: "6_finished",
        }

        # API endpoint for the checker
        self.api_url = cfg.get("vlm_api", None)

        # Whether to publish "6_finished" before exit when finish gate is True
        self.send_final_on_exit = bool(cfg.get("send_final_on_exit", True))

        # ---- Auto-success-on-repeated-failures config ----
        self.max_fail_counts: Dict[int, int] = dict(cfg.get("max_fail_counts", {}))
        self.default_max_fail_count: int = int(cfg.get("default_max_fail_count", 0))
        self.fail_counts: Dict[int, int] = {i: 0 for i in range(self.first_phase_id, self.last_phase_id + 1)}

        # ROS image reader configuration
        subscribe_right = bool(cfg.get("subscribe_right", False))
        subscribe_psm1  = bool(cfg.get("subscribe_psm1", False))
        subscribe_psm2  = bool(cfg.get("subscribe_psm2", False))
        subscribe_seg   = bool(cfg.get("subscribe_seg", False))

        self.frame_timeout       = float(cfg.get("frame_timeout_sec", 5.0))
        self.save_dir            = cfg.get("ros_frame_save_dir", "./")  # always use ./

        # Global default poll cadence; can be overridden per-task
        self.default_poll_interval_sec = float(cfg.get("poll_interval_sec", 1.0))

        # Optional per-task overrides, e.g., {1: 3.0, 2: 5.0, ...}
        # Any non-positive value is ignored and the global default is used.
        raw_overrides = cfg.get("task_poll_intervals", {})
        self.task_poll_intervals: Dict[int, float] = {}
        for k, v in raw_overrides.items():
            try:
                tid = int(k)
                val = float(v)
                if self.first_phase_id <= tid <= self.last_phase_id and val > 0:
                    self.task_poll_intervals[tid] = val
            except Exception:
                # Ignore malformed entries
                pass

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

    # ---------- Core IO helpers ----------

    def get_current_obs(self) -> str:
        """Grab latest left endoscope frame and persist it as './dvrk_left.jpg'. Returns that path."""
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
                    print(f"[HL] Warning: could not remove temp image '{temp_path}': {repr(e_rm)}")
            print(f"[HL] Saved current frame to: {fixed_path}")
            return fixed_path
        except Exception as e:
            print(f"[HL] Failed to write fixed image '{fixed_path}' from '{temp_path}': {repr(e)}")
            raise

    def _task_id_to_command_str(self, task_id: int) -> str:
        cmd = self.task_id_to_cmd.get(int(task_id))
        if not cmd:
            raise ValueError(f"Unknown task_id={task_id}. Valid keys: {sorted(self.task_id_to_cmd.keys())}")
        return cmd

    def send_llp_command(self, task_id: int) -> str:
        """Publish a one-line text instruction for the given task_id to /instructor_prediction."""
        cmd = self._task_id_to_command_str(task_id)
        try:
            self.cmd_pub.send_instruction(cmd)
            return f"[HL->LL] Sent instruction: '{cmd}'"
        except Exception as e:
            return f"[HL->LL] Failed to send instruction '{cmd}': {repr(e)}"

    # ---------- Auto-success helpers ----------

    def _max_fails_for(self, task_id: int) -> int:
        """Return the allowed consecutive failure threshold for task_id (<=0 => disabled)."""
        return int(self.max_fail_counts.get(task_id, self.default_max_fail_count))

    def _bump_fail_and_should_autosucceed(self, task_id: int) -> bool:
        """Increment consecutive-fail counter and decide whether to auto-succeed."""
        if task_id < self.first_phase_id or task_id > self.last_phase_id:
            return False  # Only tasks 1..5 are guarded
        self.fail_counts[task_id] += 1
        max_fails = self._max_fails_for(task_id)
        if max_fails <= 0 or self.fail_counts[task_id] < max_fails:
            return False
        print(f"[HL] Auto-success: task {task_id} reached {self.fail_counts[task_id]} consecutive failed checks (threshold={max_fails}).")
        self.fail_counts[task_id] = 0
        return True

    def _reset_fail_counter(self, task_id: int) -> None:
        if task_id in self.fail_counts:
            self.fail_counts[task_id] = 0

    # ---------- Poll interval + countdown ----------

    def _poll_interval_for(self, task_id: int) -> float:
        """Return per-task poll interval if set, otherwise the global default."""
        return float(self.task_poll_intervals.get(task_id, self.default_poll_interval_sec))

    def _sleep_with_countdown(self, total_seconds: float, task_id: int) -> None:
        """Sleep with a one-second countdown printed to stdout."""
        if total_seconds <= 0:
            return
        # Whole seconds
        sec = int(total_seconds)
        # Fractional tail (if any)
        tail = max(0.0, float(total_seconds) - sec)

        for remaining in range(sec, 0, -1):
            print(f"[HL] Next check for task {task_id} in {remaining:02d}s...", end="\r", flush=True)
            time.sleep(1)
        if tail > 0:
            print(f"[HL] Next check for task {task_id} in 00s...   ", end="\r", flush=True)
            time.sleep(tail)
        # Clear the countdown line
        print(" " * 60, end="\r", flush=True)

    # ---------- Finish gate (task 6) ----------

    def _check_finish_gate_with_image(self, img_path: str) -> bool:
        """Return True if task 6 is finished according to the API checker, and print the API result."""
        try:
            pred = check_task_complete(
                image_path=img_path,
                task_id=self.final_done_id,  # 6
                api_url=self.api_url
            )
            print(f"[API] task {self.final_done_id} (finish gate) complete? {bool(pred)}")
        except Exception as e:
            print(f"[HL] check_task_complete error at finish gate (task_id=6): {repr(e)}")
            return False
        return bool(pred)

    # ---------- Main loop ----------

    def run(self) -> None:
        """
        - Publish command for the starting task (configurable).
        - Each tick: sleep with countdown -> capture image -> check CURRENT TASK.
          If task is complete, advance and (if moving from 1->2) check FINISH GATE (task 6) first:
              - Capture a fresh image and check 6; if finished, optionally publish "6_finished" and EXIT.
              - Otherwise publish the next task's command and continue.
          Wrap 5->1 and repeat.

        - If a task 1..5 fails verification K times consecutively (configurable), auto-succeed it.
        """
        try:
            # Publish the very first command (start task)
            print(self.send_llp_command(self.current_task_id))

            while True:
                # Sleep with a visible countdown using the appropriate interval for the *current* task
                self._sleep_with_countdown(self._poll_interval_for(self.current_task_id), self.current_task_id)

                # Capture latest image for checking current task
                try:
                    img_path = self.get_current_obs()
                except Exception as e:
                    print(f"[HL] Failed to capture ROS image at task_id={self.current_task_id}: {repr(e)}")
                    if self._bump_fail_and_should_autosucceed(self.current_task_id):
                        is_complete = True
                    else:
                        print(self.send_llp_command(self.current_task_id))
                        print("[HL] Re-sent current command (image capture failure).")
                        continue
                else:
                    try:
                        is_complete = check_task_complete(
                            image_path=img_path,
                            task_id=self.current_task_id,
                            api_url=self.api_url
                        )
                        print(f"[API] task {self.current_task_id} complete? {bool(is_complete)}")
                    except Exception as e:
                        print(f"[HL] check_task_complete error at task_id={self.current_task_id}: {repr(e)}")
                        if self._bump_fail_and_should_autosucceed(self.current_task_id):
                            is_complete = True
                        else:
                            print(self.send_llp_command(self.current_task_id))
                            print("[HL] Re-sent current command due to verification error.")
                            continue

                if is_complete:
                    prev_task = self.current_task_id
                    self._reset_fail_counter(prev_task)
                    self.current_task_id += 1

                    # If we just completed TASK 1, check finish gate BEFORE publishing task 2
                    if prev_task == 1:
                        try:
                            finish_img = self.get_current_obs()
                        except Exception as e:
                            print(f"[HL] Failed to capture image for finish gate (after task 1): {repr(e)}")
                        else:
                            if self._check_finish_gate_with_image(finish_img):
                                if self.send_final_on_exit:
                                    print(self.send_llp_command(self.final_done_id))  # "6_finished"
                                print("[HL] Finish gate confirmed after completing task 1. Exiting.")
                                return

                    # Normal publishing of next command or wrap
                    if self.current_task_id <= self.last_phase_id:
                        print(self.send_llp_command(self.current_task_id))
                    else:
                        self.current_task_id = self.first_phase_id
                        print(self.send_llp_command(self.current_task_id))

                else:
                    # Not complete -> increment fail counter and maybe auto-succeed immediately
                    if self._bump_fail_and_should_autosucceed(self.current_task_id):
                        prev_task = self.current_task_id
                        self._reset_fail_counter(prev_task)
                        self.current_task_id += 1

                        if prev_task == 1:
                            try:
                                finish_img = self.get_current_obs()
                            except Exception as e:
                                print(f"[HL] Failed to capture image for finish gate (after task 1 auto-success): {repr(e)}")
                            else:
                                if self._check_finish_gate_with_image(finish_img):
                                    if self.send_final_on_exit:
                                        print(self.send_llp_command(self.final_done_id))
                                    print("[HL] Finish gate confirmed after auto-success of task 1. Exiting.")
                                    return

                        if self.current_task_id <= self.last_phase_id:
                            print(self.send_llp_command(self.current_task_id))
                        else:
                            self.current_task_id = self.first_phase_id
                            print(self.send_llp_command(self.current_task_id))
                    else:
                        print(self.send_llp_command(self.current_task_id))
                        print(f"[HL] Phase {self.current_task_id} not complete; re-sent current command.")

        except KeyboardInterrupt:
            print("\n[HL] Interrupted by user. Shutting down gracefully.")
        except Exception as e:
            print(f"\n[HL] Fatal error: {repr(e)}")
            raise
        finally:
            print("[HL] Controller stopped.")


# ---------- Entry point with CLI args ----------

def _parse_task_intervals(raw: str) -> Dict[int, float]:
    """
    Parse comma-separated key=value pairs into {task_id: seconds}.
    Example: "1=2.5,2=5,4=1"
    """
    if not raw:
        return {}
    out: Dict[int, float] = {}
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise argparse.ArgumentTypeError(f"Invalid task interval '{tok}'. Expected format like '3=2.5'.")
        k, v = tok.split("=", 1)
        try:
            tid = int(k.strip())
            sec = float(v.strip())
            out[tid] = sec
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid key/value in '{tok}'. Use integers for task id and number for seconds.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop ROS controller (finish check after task 1 complete)")
    parser.add_argument("--poll-interval-sec", type=float, default=55.0,
                        help="Global default polling interval in seconds (loop cadence).")
    parser.add_argument("--start-task", type=int, default=1,
                        help="Task id to start from (1..5).")
    parser.add_argument("--task-poll-intervals", type=_parse_task_intervals, default={},
                        help="Comma-separated per-task poll intervals, e.g. '1=2.5,2=5,4=1'.")
    args = parser.parse_args()

    cfg = {
        # 8880 is original model
        # 8881 is new model1
        # 8883 is new model3
        "vlm_api": "http://10.162.34.47:8881",
        "send_final_on_exit": True,

        # ROS image reader / publisher settings
        "ros_node_name": "image_reader_closed_loop",
        "ros_cmd_node_name": "instructor_publisher",

        "subscribe_right": False,
        "subscribe_psm1": False,
        "subscribe_psm2": False,
        "subscribe_seg": False,

        "frame_timeout_sec": 5.0,

        # Always save images to current directory
        "ros_frame_save_dir": "./",

        # Poll cadence (global + per-task)
        "poll_interval_sec": args.poll_interval_sec,
        "task_poll_intervals": args.task_poll_intervals,

        # Starting subtask
        "start_task_id": args.start_task,

        # ---- Auto-success configuration examples ----
        # "default_max_fail_count": 3,
        "max_fail_counts": {1: 3, 2: 3, 3: 5, 4: 2, 5: 2},
    }

    controller = ClosedLoopController(cfg)
    controller.run()
