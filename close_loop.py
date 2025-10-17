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
    After publishing phase 5's command, check task 6 (finish gate).
    If finished, optionally publish "6_finished" and exit.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # Phases 1..5; 6 is the "finish gate"
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

        # API endpoint for the checker
        self.api_url = cfg.get("vlm_api", None)

        # Whether to publish "6_finished" before exit when finish gate is True
        self.send_final_on_exit = bool(cfg.get("send_final_on_exit", True))

        # ROS image reader configuration
        subscribe_right = bool(cfg.get("subscribe_right", False))
        subscribe_psm1  = bool(cfg.get("subscribe_psm1", False))
        subscribe_psm2  = bool(cfg.get("subscribe_psm2", False))
        subscribe_seg   = bool(cfg.get("subscribe_seg", False))

        self.frame_timeout     = float(cfg.get("frame_timeout_sec", 5.0))
        self.save_dir          = cfg.get("ros_frame_save_dir", "./")  # always use ./
        self.poll_interval_sec = float(cfg.get("poll_interval_sec", 1.0))

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
        """Grab the latest left endoscope frame and persist it as './dvrk_left.jpg'. Returns that path."""
        got = self.reader.wait_for("left", timeout_sec=self.frame_timeout)
        if got is None:
            raise TimeoutError(
                f"No image received from /jhu_daVinci/left/image_raw within {self.frame_timeout} seconds."
            )

        # Save using the reader (likely a timestamped path), then copy/overwrite to fixed filename.
        temp_path  = self.reader.save("left", save_dir=self.save_dir)
        fixed_path = os.path.join(self.save_dir or "./", "dvrk_left.jpg")

        try:
            # Overwrite fixed path with the latest image
            if os.path.abspath(temp_path) != os.path.abspath(fixed_path):
                shutil.copyfile(temp_path, fixed_path)
                # Remove the temp file so that only dvrk_left.jpg remains on disk
                try:
                    os.remove(temp_path)
                except Exception as e_rm:
                    print(f"[HL] Warning: could not remove temp image '{temp_path}': {repr(e_rm)}")
            print(f"[HL] Saved current frame to: {fixed_path}")
            return fixed_path
        except Exception as e:
            print(f"[HL] Failed to write fixed image '{fixed_path}' from '{temp_path}': {repr(e)}")
            # As a last resort, still try to use the temp_path (but this violates 'only dvrk_left.jpg')
            # Prefer to re-raise to avoid leaving stray files; caller can retry next tick.
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

    # ---------- Finish gate (task 6) ----------

    def _check_finish_gate_with_image(self, img_path: str) -> bool:
        """Return True if task 6 is finished according to the API checker. Also print the API result."""
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
        - Publish command for task 1.
        - Each tick: capture image -> check current task. Print API result each time.
          If complete: advance and publish next command.
          After publishing phase 5's command (including re-sends), capture a fresh image and check task 6.
          If task 6 is finished, optionally publish "6_finished" and EXIT.
          Otherwise continue cycling 1..5.
        """
        try:
            # Publish the very first command (task 1)
            print(self.send_llp_command(self.current_task_id))

            while True:
                time.sleep(self.poll_interval_sec)

                # Capture one image per tick for checking current task
                try:
                    img_path = self.get_current_obs()
                except Exception as e:
                    print(f"[HL] Failed to capture ROS image at task_id={self.current_task_id}: {repr(e)}")
                    continue

                # Evaluate current task (1..5) and PRINT the API result
                try:
                    is_complete = check_task_complete(
                        image_path=img_path,
                        task_id=self.current_task_id,
                        api_url=self.api_url
                    )
                    print(f"[API] task {self.current_task_id} complete? {bool(is_complete)}")
                except Exception as e:
                    print(f"[HL] check_task_complete error at task_id={self.current_task_id}: {repr(e)}")
                    # Re-publish current command to keep LLP engaged
                    print(self.send_llp_command(self.current_task_id))
                    print("[HL] Re-sent current command due to verification error.")
                    continue

                if is_complete:
                    # Advance to next phase
                    self.current_task_id += 1

                    if self.current_task_id <= self.last_phase_id:
                        # Publish the next phase (2..5)
                        print(self.send_llp_command(self.current_task_id))

                        # If we just published phase 5, check finish gate now
                        if self.current_task_id == self.last_phase_id:
                            try:
                                finish_img = self.get_current_obs()
                            except Exception as e:
                                print(f"[HL] Failed to capture image for finish gate: {repr(e)}")
                                continue

                            if self._check_finish_gate_with_image(finish_img):
                                if self.send_final_on_exit:
                                    print(self.send_llp_command(self.final_done_id))  # "6_finished"
                                print("[HL] Finish gate confirmed after publishing phase 5. Exiting.")
                                break
                    else:
                        # Completed phase 5 previously; wrap back to phase 1 and publish it
                        self.current_task_id = self.first_phase_id
                        print(self.send_llp_command(self.current_task_id))
                else:
                    # Not complete -> resend current phase command
                    print(self.send_llp_command(self.current_task_id))
                    print(f"[HL] Phase {self.current_task_id} not complete; re-sent current command.")

                    # If we are (re-)publishing phase 5, also check finish gate after the resend
                    if self.current_task_id == self.last_phase_id:
                        try:
                            finish_img = self.get_current_obs()
                        except Exception as e:
                            print(f"[HL] Failed to capture image for finish gate (resend path): {repr(e)}")
                            continue

                        if self._check_finish_gate_with_image(finish_img):
                            if self.send_final_on_exit:
                                print(self.send_llp_command(self.final_done_id))  # "6_finished"
                            print("[HL] Finish gate confirmed after re-publishing phase 5. Exiting.")
                            break

        except KeyboardInterrupt:
            print("\n[HL] Interrupted by user. Shutting down gracefully.")
        except Exception as e:
            print(f"\n[HL] Fatal error: {repr(e)}")
            raise
        finally:
            print("[HL] Controller stopped.")


# ---------- Entry point with CLI arg for poll interval ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop ROS controller (no human verification)")
    parser.add_argument("--poll-interval-sec", type=float, default=10.0,
                        help="Polling interval in seconds (loop cadence).")
    args = parser.parse_args()

    cfg = {
        "vlm_api": "http://10.162.34.47:8888",
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

        # Poll cadence
        "poll_interval_sec": args.poll_interval_sec,
    }

    controller = ClosedLoopController(cfg)
    controller.run()
