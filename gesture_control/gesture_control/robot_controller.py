# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from geometry_msgs.msg import Twist

# class RobotController(Node):
#     def __init__(self):
#         super().__init__('robot_controller')
#         self.subscription = self.create_subscription(
#             String, 'gesture_command', self.command_callback, 10)
#         self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

#     def command_callback(self, msg):
#         """Callback function for processing gesture commands."""
#         cmd = Twist()
        
#         if msg.data == "Move Forward":
#             cmd.linear.x = 0.5  # Move forward
#         elif msg.data == "Stop":
#             cmd.linear.x = 0.0  # Stop
#         elif msg.data == "Turn Left":
#             cmd.angular.z = 0.5  # Rotate left
#         elif msg.data == "Turn Right":
#             cmd.angular.z = -0.5  # Rotate right

#         self.publisher.publish(cmd)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RobotController()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class TurtleController(Node):
    def __init__(self):
        super().__init__('turtle_controller')
        self.subscription = self.create_subscription(
            String, 'gesture_command', self.command_callback, 10)
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

    def command_callback(self, msg):
        """Callback function for processing gesture commands."""
        cmd = Twist()

        if msg.data == "Move Forward":
            cmd.linear.x = 2.0  # Avancer
        elif msg.data == "Stop":
            cmd.linear.x = 0.0  # Arrêt
        elif msg.data == "Turn Left":
            cmd.angular.z = 2.0  # Tourner à gauche
        elif msg.data == "Turn Right":
            cmd.angular.z = -2.0  # Tourner à droite

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
