#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal HL→LL publisher for SRT-H

- Publishes a single-line text instruction to `/instructor_prediction` (std_msgs/String)
- Matches LLP subscriber:
    self.instruction_sub = rospy.Subscriber(
        "/instructor_prediction", String, self.language_instruction_callback, queue_size=10
    )
"""

import rospy
from std_msgs.msg import String

class InstructorCommandPublisher(object):
    """
    Simple publisher that sends exactly one text command per call
    to the `/instructor_prediction` topic (std_msgs/String).
    """

    def __init__(self, node_name="instructor_publisher",
                 instruction_topic="/instructor_prediction",
                 queue_size=10):
        # Initialize node if not already initialized by a larger app
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        # Only publisher required by LLP
        self.instruction_pub = rospy.Publisher(
            instruction_topic, String, queue_size=queue_size
        )

        # Give ROS some time to set up connections (optional but helpful)
        rospy.sleep(0.2)

    def wait_for_subscribers(self, timeout_sec=2.0):
        """
        Optionally wait for at least one subscriber to connect
        to reduce the chance of dropping the first message.
        """
        start = rospy.Time.now()
        rate = rospy.Rate(50)
        while self.instruction_pub.get_num_connections() == 0:
            if (rospy.Time.now() - start).to_sec() > timeout_sec:
                return False
            rate.sleep()
        return True

    def send_instruction(self, text_command):
        """
        Publish one line of text to `/instructor_prediction`.

        Args:
            text_command (str): e.g., "1_retract_home_in"
        """
        if not isinstance(text_command, str) or not text_command.strip():
            raise ValueError("text_command must be a non-empty string.")

        # (Optional) ensure a subscriber is listening
        self.wait_for_subscribers(timeout_sec=2.0)

        # Publish the primary instruction
        self.instruction_pub.publish(String(data=text_command))
        rospy.loginfo(f"[HL->LL] Sent instruction='{text_command}'")

if __name__ == "__main__":
    """
    Example:
        rosrun your_pkg instructor_publisher.py
    """
    try:
        pub = InstructorCommandPublisher()
        # Example: send one instruction, then exit
        pub.send_instruction("1_retract_home_in")
        # If you want to keep the node alive, uncomment:
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass