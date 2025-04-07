import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import cv2
import numpy as np
import struct
import sys

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

def blur(image1, n):
    image = image1.copy()
    ni = int(image.shape[0]//n-1)
    nj = int(image.shape[1]//n-1)

    for i in range(ni):
        for j in range(nj):
            if len(image.shape) == 2:
                pixel_sum = 0
            else:
                pixel_sum = np.zeros(int(image.shape[-1]))
            
            # count = 0
            # for i1 in range(n*i,n*(i+1)):
            #     for j1 in range(n*j,n*(j+1)):
            #         pixel_sum = (pixel_sum*count + image[i1,j1])/(count+1)
                    
            # count += 1
            try:
                for i1 in range(3):
                    image[n*i:n*(i+1),n*j:n*(j+1)] = np.int8(np.sum(image[n*i:n*(i+1),n*j:n*(j+1)])/n**2)
            except:
                x34 = 3
    return image


######################

class ColorObjDetectionNode(Node):
    def __init__(self):
        super().__init__('color_goal_detection_node')
        self.get_logger().info('Color Goal Detection Node Started')
        
        # Declare the parameters for the color detection
        self.declare_parameter('color_low', [0, 150, 50])
        self.declare_parameter('color_high', [130, 255, 255])
        self.declare_parameter('object_size_min', 1000)
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        self.searching = True
        self.first = True
        self.count = 1000
        self.center_x = 30
        self.center_y = 40
        self.located = False
        
        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the detected object and the bounding box
        self.pub_detected_obj = self.create_publisher(Image, '/detected_color_goal',10)
        self.pub_detected_obj2 = self.create_publisher(Image, '/detected_color_goal_map',10)
        self.pub_detected_obj_pose = self.create_publisher(PoseStamped, '/detected_color_goal_pose', 10)
        # Create a subscriber to the RGB and Depth images
        self.sub_rgb = Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_dep = Subscriber(self, Image, '/camera/depth/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth/points')
        # Create a time synchronizer
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_dep], 10, 0.1)
        # Register the callback to the time synchronizer
        self.ts.registerCallback(self.camera_callback)

    def camera_callback(self, rgb_msg, points_msg, dep_msg):
        # self.get_logger().info('Received RGB and Depth Messages')
        # get ROS parameters
        # param_color_low = np.array(self.get_parameter('color_low').value)
        # param_color_high = np.array(self.get_parameter('color_high').value)
        param_object_size_min = self.get_parameter('object_size_min').value
        
        # self.get_logger().info('Color Low: {}'.format(param_color_low))
        # self.get_logger().info('Color High: {}'.format(param_color_high))
        if self.searching:
            if self.first:
                self.past_image = self.br.imgmsg_to_cv2(rgb_msg,"bgr8")
                self.get_logger().info('Area Identified')
                self.first = False

                # Greyscale Area Image
                # plt.imshow(im_age1,'gray')
                # plt.savefig('im_age1.png')
                # plt.close()

            else:
                self.current_image = self.br.imgmsg_to_cv2(rgb_msg,"bgr8")

                im_age1 = cv2.cvtColor(self.past_image, cv2.COLOR_BGR2GRAY)
                im_age2 = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                
                ni = 5
                # self.get_logger().info('Item Identified: {}'.format(ni))
                
                im_age2 = blur(im_age2, 5)
                im_age1 = blur(im_age1, 5)
                im_age3 = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
                
                # Greyscale Camera Image
                # plt.imshow(im_age2,'gray')
                # plt.savefig('im_age2.png')
                # plt.close()

                image = im_age1.copy()
                
                img_index = np.where(im_age1[:,:]>im_age2[:,:]) 
                image[img_index[0],img_index[1]] = im_age1[img_index[0],img_index[1]] - im_age2[img_index[0],img_index[1]]

                img_index = np.where(im_age1[:,:]<=im_age2[:,:]) 

                image[img_index[0],img_index[1]] = im_age2[img_index[0],img_index[1]] - im_age1[img_index[0],img_index[1]]

                #Grayscale Difference between Images
                plt.imshow(image,'gray')
                plt.savefig('grayscale_difference0.png')
                plt.close()

                con = np.max(image)*.5
                self.get_logger().info('Item Identified, Difference Floor: {}'.format(con))
                if con < 50:
                    return
                
                index1 = np.where(image<=con)
                image[index1[0],index1[1]]=0
                    
                # #Grayscale difference, No background
                # plt.imshow(image,'gray')
                # plt.savefig('grayscale_difference1.png')
                # plt.close()

                index1 = np.where(image>con)                    
                image[index1[0],index1[1]]=255

                #Grayscale difference, White and Black
                plt.imshow(image,'gray')
                plt.savefig('grayscale_difference2.png')
                plt.close()

                try:
                    self.get_logger().info('Try Contour')

                    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) > 0:
                        self.get_logger().info('Item Identified: {}'.format(len(contours)))
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        self.get_logger().info('Item Identified: {}'.format(w*h))
                        if w * h < param_object_size_min:
                            return
                        # threshold by size    
                        # draw rectangle
                        image=cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0),5)
                        
                        ##Grayscale difference, Black and White, With box
                        plt.imshow(image,'gray')
                        plt.savefig('rgb_image_goal_box.png')
                        plt.close()

                        center_x = int(x + w/2)
                        center_y = int(y + h/2)
                    
                        self.current_image = self.br.imgmsg_to_cv2(rgb_msg,"bgr8")
                        im_age3 = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
                        pixed_image = im_age3

                        # 
                        
                        # pixed_image[:,:] = pixed_image[center_x,center_y]

                        color = pixed_image[center_y,center_x]
                        self.get_logger().info('Item Identified: {}'.format(color))
                        
                        self.count+=1
                        self.param_color_low = np.zeros(3)

                        self.param_color_high = np.ones(3)*255
                        self.param_color_high[0] = np.array(color)[0]
                        self.param_color_high = np.array(color)+10.
                        self.param_color = np.array(color)

                        pixed_image = pixed_image[int(center_y-h*.1):int(center_y+h*.1),int(center_x-w*.1):int(center_x)]

                        #Parameter Generation Image
                        plt.imshow(pixed_image,'hsv')
                        plt.savefig('hsv_color_picker2.png')
                        plt.close()
                        
                        colors_all = np.zeros(int(pixed_image.shape[0]*pixed_image.shape[1]))
                        for id1 in range(3):
                            in2 = 0
                            for tow in pixed_image:
                                for seet in tow:
                                    colors_all[in2] = seet[id1]
                                    in2+=1
                            colors_all = np.sort(colors_all)
                            self.param_color_low[id1] = colors_all[10]
                            self.param_color_high[id1] = colors_all[-10]
                        
                        #Parameter Color High
                        pixed_image_high = pixed_image.copy()
                        pixed_image_high[:,:] = self.param_color_high
                        plt.imshow(cv2.cvtColor(pixed_image_high, cv2.COLOR_HSV2RGB))
                        plt.savefig('color_high.png')
                        plt.close()

                        #Parameter Color Low
                        pixed_image_low = pixed_image.copy()
                        pixed_image_low[:,:] = self.param_color_low
                        plt.imshow(cv2.cvtColor(pixed_image_low, cv2.COLOR_HSV2RGB))
                        plt.savefig('color_low.png')
                        plt.close()

                        self.get_logger().info('Color Low: {}'.format(self.param_color_low))
                        self.get_logger().info('Color High: {}'.format(self.param_color_high))

                        if self.located:
                            self.searching = False
                        self.located = True
                        
                except Exception as e:
                    self.get_logger().error('Error: {}'.format(e))
                    return

        else:
                
            # Convert the ROS image message to a numpy array
            rgb_image = self.br.imgmsg_to_cv2(rgb_msg,"bgr8")

            #Camera image
            # plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            # plt.savefig('rgb_image_goal.png')
            # plt.close()
            
            # to hsv
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

            #HSV of Camera Image
            # plt.imshow(hsv_image,'hsv')
            # plt.savefig('hsv_image_goal.png')
            # plt.close()
            
            # color mask
            color_mask = cv2.inRange(hsv_image, self.param_color_low, self.param_color_high)

            # #Color Mask of goal parameters
            # plt.imshow(color_mask,'gray')
            # plt.savefig('color_mask_goal.png')
            # plt.close()

            # find largest contour
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # threshold by size
                # if w * h < param_object_size_min:
                #    return

                # draw rectangle
                rgb_image=cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                
            else:
                self.get_logger().info('No Contours')
                return
            
            # #Detected Image plot before Transfor
            # plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            # plt.savefig('rgb_image_goal_300.png')
            # plt.close()
            
            # get the location of the detected object using point cloud
            pointid = (center_y*points_msg.row_step) + (center_x*points_msg.point_step)
            (X, Y, Z) = struct.unpack_from('fff', points_msg.data, offset=pointid)
            center_points = np.array([X,Y,Z])

            if np.any(np.isnan(center_points)):
                return
            

            try:
                # Transform the center point from the camera frame to the world frame
                transform = self.tf_buffer.lookup_transform('base_footprint',rgb_msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.2))
                #transform = self.tf_buffer.lookup_transform('world_frame_id',rgb_msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.2))
                
                t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
                cp_robot = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
                
                # Create a pose message for the detected object
                detected_obj_pose = PoseStamped()
                detected_obj_pose.header.frame_id = 'base_footprint'
                detected_obj_pose.header.stamp = rgb_msg.header.stamp
                detected_obj_pose.pose.position.x = cp_robot[0]
                detected_obj_pose.pose.position.y = cp_robot[1]
                detected_obj_pose.pose.position.z = cp_robot[2]
            except TransformException as e:
                self.get_logger().error('Transform Error: {}'.format(e))
                return
            
            # Publish the detected object
            self.pub_detected_obj_pose.publish(detected_obj_pose)

            # publush the detected object image
            detect_img_msg = self.br.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            detect_img_msg.header = rgb_msg.header
            self.get_logger().info('image message published')
            self.pub_detected_obj.publish(detect_img_msg)
            
            #Current Detected Goal object
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            plt.savefig('rgb_image_rect_goal.png')
            plt.close()

            ##All Detected Goal objects
            #plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            #plt.savefig('rgb_image_rect_goal' + str(self.count)+'.png')
            #plt.close()

            self.count += 1
        return
def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    color_obj_detection_node = ColorObjDetectionNode()
    # Spin the node so the callback function is called.
    rclpy.spin(color_obj_detection_node)
    # Destroy the node explicitly
    color_obj_detection_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()