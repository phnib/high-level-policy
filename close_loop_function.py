import time
from typing import Tuple, Dict, Any

from model import check_task_complete

# These two come from ROS utilities
from ros_image_reader import ROSImageReader
from ros_command_publisher import InstructorCommandPublisher


class ClosedLoopController:
    """Closed-loop high-level controller without any LLM involvement."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

        # Phases 1..5, then "6_finished"
        self.current_task_id = 1
        self.first_phase_id = 1
        self.last_phase_id = 5     # per requirement: 5 phases
        self.final_done_id = 6     # publish "6_finished" after phase 5

        # Command mapping (kept consistent with your previous setup)
        self.task_id_to_cmd = {
            1: "1_retract_home_in",
            2: "2_resect_home_in",
            3: "3_resect",
            4: "4_resect_home_out",
            5: "5_retract_home_out",
            6: "6_finished",
        }

        # API endpoint kept for compatibility (your ResNet-backed checker can ignore or use it)
        self.api_url = cfg.get("vlm_api", None)

        # ROS image reader configuration
        subscribe_right = bool(cfg.get("subscribe_right", False))
        subscribe_psm1 = bool(cfg.get("subscribe_psm1", False))
        subscribe_psm2 = bool(cfg.get("subscribe_psm2", False))
        subscribe_seg = bool(cfg.get("subscribe_seg", False))

        self.frame_timeout = float(cfg.get("frame_timeout_sec", 5.0))
        self.save_dir = cfg.get("ros_frame_save_dir", None)
        self.poll_interval_sec = float(cfg.get("poll_interval_sec", 1.0))  # 1s by requirement

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
        """Grab the latest left endoscope frame and persist it. Returns local path."""
        got = self.reader.wait_for("left", timeout_sec=self.frame_timeout)
        if got is None:
            raise TimeoutError(
                f"No image received from /jhu_daVinci/left/image_raw within {self.frame_timeout} seconds."
            )
        out_path = self.reader.save("left", save_dir=self.save_dir)
        return out_path

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

    # ---------- Human-in-the-loop (same behavior/UX as before) ----------

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

        # Collect corrected values
        while True:
            task_str = input("Please enter the CORRECT task_id (integer): ").strip()
            try:
                corrected_task_id = int(task_str)
                break
            except Exception:
                print("Invalid task_id. Please input an integer.")

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

    # ---------- Main loop ----------

    def run(self) -> None:
        """
        Closed-loop run:
          - publish phase 1 command
          - loop at 1 Hz: capture image -> check_task_complete -> human verify
          - if confirmed complete => advance and publish next phase
          - after phase 5 is complete => publish "6_finished" and exit
        """
        try:
            # Publish the very first command
            print(self.send_llp_command(self.current_task_id))

            while True:
                # If we already completed phase 5, send "6_finished" and break
                if self.current_task_id > self.last_phase_id:
                    print(self.send_llp_command(self.final_done_id))
                    print("[HL] All phases complete. Exiting.")
                    break

                # Poll at the fixed interval
                time.sleep(self.poll_interval_sec)

                # 1) Capture current observation
                try:
                    img_path = self.get_current_obs()
                except Exception as e:
                    print(f"[HL] Failed to capture ROS image at task_id={self.current_task_id}: {repr(e)}")
                    continue  # try again next tick

                # 2) Call the (ResNet-backed) completion checker with unchanged interface
                try:
                    is_complete = check_task_complete(
                        image_path=img_path,
                        task_id=self.current_task_id,
                        api_url=self.api_url
                    )
                except Exception as e:
                    print(f"[HL] check_task_complete error at task_id={self.current_task_id}: {repr(e)}")
                    continue  # try again next tick

                # 3) Human verification (same UX as before)
                final_task_id, final_bool = self.human_verify_vlm_output(
                    predicted_task_id=self.current_task_id,
                    predicted_bool=is_complete
                )

                # Align internal pointer if human corrected the task id
                if final_task_id != self.current_task_id:
                    # Clamp within known range [1, 6] to be safe
                    final_task_id = max(self.first_phase_id, min(self.final_done_id, final_task_id))
                    self.current_task_id = final_task_id

                # 4) If confirmed complete -> move to the next phase and publish its command
                if final_bool is True:
                    self.current_task_id += 1
                    if self.current_task_id <= self.last_phase_id:
                        print(self.send_llp_command(self.current_task_id))
                    else:
                        # Will publish "6_finished" at the top of the loop in the next iteration
                        pass
                else:
                    # Not complete yet; keep polling within the same phase
                    print(self.send_llp_command(self.current_task_id))
                    print(f"[HL] Phase {self.current_task_id} not complete, send the complete command.")

        except KeyboardInterrupt:
            print("\n[HL] Interrupted by user. Shutting down gracefully.")
        except Exception as e:
            print(f"\n[HL] Fatal error: {repr(e)}")
        finally:
            # Nothing special to tear down here; ROS nodes are managed by the helper classes.
            print("[HL] Controller stopped.")


# ---------- Optional: simple entry point ----------

if __name__ == "__main__":
    # Minimal inline config; replace or extend with your own config source.
    cfg = {
        # ResNet checker can ignore or use this; kept for backward compatibility
        "vlm_api": 'http://10.162.34.47:8888',

        # ROS image reader / publisher settings
        "ros_node_name": "image_reader_closed_loop",
        "ros_cmd_node_name": "instructor_publisher",

        "subscribe_right": False,
        "subscribe_psm1": False,
        "subscribe_psm2": False,
        "subscribe_seg": False,

        "frame_timeout_sec": 5.0,
        "ros_frame_save_dir": None,      # or a path to persist frames
        "poll_interval_sec": 1.0,        # 1 second by requirement
    }

    controller = ClosedLoopController(cfg)
    controller.run()
