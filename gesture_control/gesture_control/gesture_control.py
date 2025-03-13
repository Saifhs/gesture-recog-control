# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# import cv2
# import mediapipe as mp

# class GestureControlNode(Node):
#     def __init__(self):
#         super().__init__('gesture_control')
#         self.publisher = self.create_publisher(String, 'gesture_command', 10)
        
#         # Initialize MediaPipe Hands
#         self.mp_hands = mp.solutions.hands
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#         self.cap = cv2.VideoCapture(0)
#         self.run_gesture_detection()

#     def recognize_gesture(self, landmarks):
#         """Identifies the gesture based on finger positions."""
#         thumb_tip = landmarks[4][1]   # Thumb
#         index_tip = landmarks[8][1]   # Index finger
#         middle_tip = landmarks[12][1] # Middle finger
#         ring_tip = landmarks[16][1]   # Ring finger
#         pinky_tip = landmarks[20][1]  # Little finger

#         # Open Palm -> Move Forward
#         if index_tip < landmarks[6][1] and middle_tip < landmarks[10][1] and ring_tip < landmarks[14][1] and pinky_tip < landmarks[18][1]:
#             return "Move Forward"

#         # Fist -> Stop
#         if index_tip > landmarks[6][1] and middle_tip > landmarks[10][1] and ring_tip > landmarks[14][1] and pinky_tip > landmarks[18][1]:
#             return "Stop"

#         # Thumb Left -> Turn Left
#         if thumb_tip < index_tip and thumb_tip < middle_tip:
#             return "Turn Left"

#         # Thumb Right -> Turn Right
#         elif thumb_tip > index_tip and thumb_tip > middle_tip:
#             return "Turn Right"

#         return "Unknown"

#     def run_gesture_detection(self):
#         """Runs the webcam feed and processes gestures."""
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # Convert frame to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = self.hands.process(frame_rgb)

#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
#                     gesture = self.recognize_gesture(landmarks)

#                     # Publish gesture as a ROS2 message
#                     msg = String()
#                     msg.data = gesture
#                     self.publisher.publish(msg)

#                     # Draw landmarks
#                     self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#                     cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             cv2.imshow("Gesture Recognition", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()

# def main(args=None):
#     rclpy.init(args=args)
#     node = GestureControlNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control')
        self.publisher = self.create_publisher(String, 'gesture_command', 10)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure the pipeline to stream depth and color frames
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            # Start the pipeline
            self.pipeline.start(self.config)
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start pipeline: {e}")
            return

        self.run_gesture_detection()

    def recognize_gesture(self, landmarks):
        """Identifies the gesture based on finger positions."""
        thumb_tip = landmarks[4][1]   # Thumb
        index_tip = landmarks[8][1]   # Index finger
        middle_tip = landmarks[12][1] # Middle finger
        ring_tip = landmarks[16][1]   # Ring finger
        pinky_tip = landmarks[20][1]  # Little finger

        # Open Palm -> Move Forward
        if index_tip < landmarks[6][1] and middle_tip < landmarks[10][1] and ring_tip < landmarks[14][1] and pinky_tip < landmarks[18][1]:
            return "Move Forward"

        # Fist -> Stop
        if index_tip > landmarks[6][1] and middle_tip > landmarks[10][1] and ring_tip > landmarks[14][1] and pinky_tip > landmarks[18][1]:
            return "Stop"

        # Thumb Left -> Turn Left
        if thumb_tip < index_tip and thumb_tip < middle_tip:
            return "Turn Left"

        # Thumb Right -> Turn Right
        elif thumb_tip > index_tip and thumb_tip > middle_tip:
            return "Turn Right"

        return "Unknown"

    def run_gesture_detection(self):
        """Runs the RealSense feed and processes gestures."""
        try:
            while True:
                # Wait for a frame from the RealSense camera
                frames = self.pipeline.wait_for_frames()

                # Get color frame
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert RealSense frame to OpenCV format
                frame = np.asanyarray(color_frame.get_data())

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                        gesture = self.recognize_gesture(landmarks)

                        # Publish gesture as a ROS2 message
                        msg = String()
                        msg.data = gesture
                        self.publisher.publish(msg)

                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop RealSense pipeline
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()