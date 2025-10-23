import argparse
import os
import shutil
import time
from typing import Tuple, Dict, Any

from model import check_task_complete

# ROS utilities
from ros_image_reader import ROSImageReader
from ros_command_publisher import InstructorCommandPublisher


class ClosedLoopController:
    """Closed-loop high-level controller with ROS.
    Cycle: 1->2->3->4->5->(wrap)->1 ...
    After publishing the command for phase 5 (the last in the cycle),
    check task 6 (finish gate). If confirmed finished, publish "6_finished" and exit.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # Phases 1..5; 6 is only a "finish gate" signal (not part of normal cycle)
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

        # Whether to publish "6_finished" before exit when the finish gate is true
        self.send_final_on_exit = bool(cfg.get("send_final_on_exit", True))

        # ROS image reader configuration
        subscribe_right = bool(cfg.get("subscribe_right", False))
        subscribe_psm1 = bool(cfg.get("subscribe_psm1", False))
        subscribe_psm2 = bool(cfg.get("subscribe_psm2", False))
        subscribe_seg = bool(cfg.get("subscribe_seg", False))

        self.frame_timeout = float(cfg.get("frame_timeout_sec", 5.0))
        # Always save into current directory
        self.save_dir = cfg.get("ros_frame_save_dir", "./")

        # Poll interval (overridable via CLI)
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

        # Save using the reader (may produce a timestamped name), then copy/overwrite to fixed filename.
        temp_path = self.reader.save("left", save_dir=self.save_dir)
        fixed_path = os.path.join(self.save_dir or "./", "dvrk_left.jpg")

        try:
            if os.path.abspath(temp_path) != os.path.abspath(fixed_path):
                shutil.copyfile(temp_path, fixed_path)
        except Exception as e:
            print(f"[HL] Failed to write fixed image '{fixed_path}' from '{temp_path}': {repr(e)}")
            print(f"[HL] Saved current frame to: {temp_path}")
            return temp_path

        print(f"[HL] Saved current frame to: {fixed_path}")
        return fixed_path

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

    # ---------- Human-in-the-loop ----------

    def human_verify_vlm_output(self, predicted_task_id: int, predicted_bool: bool) -> Tuple[int, bool]:
        """
        Ask a human to confirm or correct the model's prediction.
        Returns (final_task_id, final_bool).
        """
        print("\n=== Human Verification ===")
        print(f"Model predicted -> task_id={predicted_task_id}, complete={predicted_bool}")
        ans = input("Is this correct? [y/N]: ").strip().lower()

        if ans in ("y", "yes"):
            print("Keeping model output.")
            return predicted_task_id, predicted_bool

        # Keep task id fixed to predicted value; ask only for boolean correction
        corrected_task_id = predicted_task_id

        while True:
            bool_str = input("Please enter the CORRECT boolean (true/false): ").strip().lower()
            if bool_str in ("true", "t", "1"):
                corrected_bool = True
                break
            elif bool_str in ("false", "f", "0"):
                corrected_bool = False
                break
            else:
                print("Invalid boolean. Please input 'true' or 'false'.")

        print(f"Human corrected -> task_id={corrected_task_id}, complete={corrected_bool}")
        return corrected_task_id, corrected_bool

    # ---------- Finish gate (task 6) after publishing phase 5 ----------

    def _check_finish_gate_with_image(self, img_path: str) -> bool:
        """
        Check task 6 (finish) using the given image; return True if confirmed finished.
        """
        try:
            pred = check_task_complete(
                image_path=img_path,
                task_id=self.final_done_id,  # 6
                api_url=self.api_url
            )
        except Exception as e:
            print(f"[HL] check_task_complete error at finish gate (task_id=6): {repr(e)}")
            return False

        # Human verification; we only care about the final boolean
        _, final_bool = self.human_verify_vlm_output(
            predicted_task_id=self.final_done_id,
            predicted_bool=pred
        )
        return bool(final_bool)

    # ---------- Main loop ----------

    def run(self) -> None:
        """
        - Immediately publish command for task 1.
        - Each tick: capture image -> check current task -> human verify.
          If confirmed complete: advance pointer and publish the next command.
          After publishing the command for phase 5 (the last in cycle), immediately check task 6.
            - If task 6 is finished, optionally publish "6_finished" and EXIT.
            - Otherwise continue cycling 1..5 forever.
        - We do NOT check task 6 at phase 1 anymore.
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

                # Evaluate current task (1..5) using the image
                try:
                    is_complete = check_task_complete(
                        image_path=img_path,
                        task_id=self.current_task_id,
                        api_url=self.api_url
                    )
                except Exception as e:
                    print(f"[HL] check_task_complete error at task_id={self.current_task_id}: {repr(e)}")
                    # Keep polling without advancing
                    print(self.send_llp_command(self.current_task_id))
                    print(f"[HL] Re-sent current command due to verification error.")
                    continue

                # Human verification for the current task
                final_task_id, final_bool = self.human_verify_vlm_output(
                    predicted_task_id=self.current_task_id,
                    predicted_bool=is_complete
                )

                # If human corrected the id, align pointer safely to [1..6] but keep our 1..5 cycling policy
                if final_task_id != self.current_task_id:
                    final_task_id = max(self.first_phase_id, min(self.final_done_id, final_task_id))
                    self.current_task_id = final_task_id

                if final_bool is True:
                    # Advance to next phase
                    self.current_task_id += 1

                    if self.current_task_id <= self.last_phase_id:
                        # Publish the next phase (2..5)
                        print(self.send_llp_command(self.current_task_id))

                        # If we just published the last phase (5), check finish gate right now.
                        if self.current_task_id == self.last_phase_id:
                            # Grab a fresh image *after* the last command to assess finish condition.
                            try:
                                finish_img = self.get_current_obs()
                            except Exception as e:
                                print(f"[HL] Failed to capture image for finish gate: {repr(e)}")
                                continue

                            finished = self._check_finish_gate_with_image(finish_img)
                            if finished:
                                if self.send_final_on_exit:
                                    print(self.send_llp_command(self.final_done_id))  # "6_finished"
                                print("[HL] Finish gate confirmed after publishing phase 5. Exiting.")
                                break  # exit loop

                    else:
                        # Completed phase 5 previously; wrap back to phase 1 and publish it
                        self.current_task_id = self.first_phase_id
                        print(self.send_llp_command(self.current_task_id))

                else:
                    # Not complete -> resend current phase command
                    print(self.send_llp_command(self.current_task_id))
                    print(f"[HL] Phase {self.current_task_id} not complete; re-sent current command.")

                    # If we are re-sending phase 5, we also check finish gate after re-publish.
                    if self.current_task_id == self.last_phase_id:
                        try:
                            finish_img = self.get_current_obs()
                        except Exception as e:
                            print(f"[HL] Failed to capture image for finish gate (resend path): {repr(e)}")
                            continue

                        finished = self._check_finish_gate_with_image(finish_img)
                        if finished:
                            if self.send_final_on_exit:
                                print(self.send_llp_command(self.final_done_id))  # "6_finished"
                            print("[HL] Finish gate confirmed after re-publishing phase 5. Exiting.")
                            break  # exit loop

        except KeyboardInterrupt:
            print("\n[HL] Interrupted by user. Shutting down gracefully.")
        except Exception as e:
            print(f"\n[HL] Fatal error: {repr(e)}")
            raise
        finally:
            print("[HL] Controller stopped.")


# ---------- Optional: simple entry point with CLI arg for poll interval ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop ROS controller")
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

        # Overridden by CLI
        "poll_interval_sec": args.poll_interval_sec,
    }

    controller = ClosedLoopController(cfg)
    controller.run()
