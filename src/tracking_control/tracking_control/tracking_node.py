import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = np.array([.1,.1,.1])
        self.goal_pose = np.array([.1,.1,.1])
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        # try:
        #     # Transform the center point from the camera frame to the world frame
        #     transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
        #     t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
        #     cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        # except TransformException as e:
        #     self.get_logger().error('Transform Error: {}'.format(e))
        #     return
        
        # Get the detected object pose in the world frame
        self.obs_pose = center_points

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        # try:
        #     # Transform the center point from the camera frame to the world frame
        #     transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
        #     t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
        #     cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        # except TransformException as e:
        #     self.get_logger().error('Transform Error: {}'.format(e))
        #     return
        
        # Get the detected object pose in the world frame
        self.goal_pose = center_points
        
    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            if not np.isscalar(self.obs_pose) and self.obs_pose.size == robot_world_R.shape[-1]:    
                obstacle_pose = robot_world_R@self.obs_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            else:
                self.get_logger().info('Transform Error Obj: {}'.format(np.isscalar(self.obs_pose)))
                self.get_logger().info('Transform Error Obj: {}'.format(np.isscalar(self.obs_pose.size)))
                self.get_logger().info('Transform Error Obj: {}'.format(np.isscalar(robot_world_R.shape[-1])))
                obstacle_pose = np.array([robot_world_x*2,robot_world_y,robot_world_z])
            if not np.isscalar(self.goal_pose) and self.goal_pose.size == robot_world_R.shape[-1]:
                goal_pose = robot_world_R@self.goal_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            else:
                self.get_logger().info('Transform Error Goal: {}'.format(np.isscalar(self.goal_pose)))
                self.get_logger().info('Transform Error Goal: {}'.format(np.isscalar(self.goal_pose.size)))
                self.get_logger().info('Transform Error Goal: {}'.format(np.isscalar(robot_world_R.shape[-1])))
                goal_pose = np.array([robot_world_x,robot_world_y*2,robot_world_z])

    
        
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return np.ones(3), np.ones(3)
        
        return obstacle_pose, goal_pose
    
    def timer_update(self):
        ################### Write your code here ###################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly
        if self.goal_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        # Get the current object pose in the robot base_footprint frame
        # current_obs_pose, current_goal_pose = self.get_current_poses()
        
        # TODO: get the control velocity command
        cmd_vel = self.controller()
        
        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########
        
        # TODO: Update the control velocity command
        # Edit on 10MAR2025 By Marilla - Sudo code in lines below
        # Have proportional controller to object in x direction
        # Implement something tangent to object in y or z direction - can move around object while still moving toward goal
        # Need current robot position, goal position, and obstacle position(s)
        # Use equation from lecture 12 slide 20
        try:
            current_obs_pose, current_goal_pose = self.get_current_poses()
        except:
            current_obs_pose, current_goal_pose = np.ones(3),np.ones(3)
        self.get_logger().info('Transform Goal: {}'.format(current_obs_pose))
        self.get_logger().info('Transform Obs: {}'.format(current_goal_pose))
        k_rep = 10
        k_att = 1
        k_turn = 1
        
        q_star = 0.5 #Check units
        obj_dist = math.sqrt((current_obs_pose[0])**2 + (current_obs_pose[1])**2)
        goal_dist = math.sqrt((current_goal_pose[0])**2 + (current_goal_pose[1])**2)
        
        # Repulsive field
        if obj_dist < q_star:
            grad_dist_x =  current_obs_pose[0]/ obj_dist
            grad_dist_y = current_obs_pose[1]/ obj_dist
            u_x_rep = k_rep*(1/q_star - 1/obj_dist)*1/(obj_dist**2)*grad_dist_x
            u_y_rep = k_rep*(1/q_star - 1/obj_dist)*1/(obj_dist**2)*grad_dist_y
        else:
            u_x_rep = 0.0
            u_y_rep = 0.0
        
        # Attractive field in x direction (drive straight forward to goal
        if goal_dist > 0.3:
            u_x_att = k_att*current_goal_pose[0]
        else:
            u_x_att = 0.0
        
        # Turn towards goal
        u_z_att = -k_turn*current_goal_pose[1]

        cmd_vel = Twist()
        cmd_vel.linear.x = u_x_att
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = u_z_att
        # self.get_logger().info('Transform Error: {}'.format(cmd_vel))
        return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
