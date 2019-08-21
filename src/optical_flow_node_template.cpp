#include <ros/ros.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include "visualization_msgs/Marker.h"

using namespace std;
double height;
double vicon_time, height_time, last_vicon_time, image_time;
Eigen::Vector3d velocity, position, last_position, velocity_gt;
Eigen::Quaterniond q;
ros::Publisher pub_vel, pub_vel_gt;
cv::Mat tmp, image;
double fx, fy, cx, cy;
cv::Mat cameraMatrix, distCoeffs;
cv::Size imageSize;

void visualizeVelocity(Eigen::Vector3d position, Eigen::Vector3d velocity,
                       int id, Eigen::Vector3d color, ros::Publisher pub_vel) {
    double scale = 10;
    visualization_msgs::Marker m;
    m.header.frame_id = "world";
    m.id = id;
    m.type = visualization_msgs::Marker::ARROW;
    m.action = visualization_msgs::Marker::MODIFY;
    m.scale.x = 0.2;
    m.scale.y = 0.5;
    m.scale.z = 0;
    m.pose.position.x = 0;
    m.pose.position.y = 0;
    m.pose.position.z = 0;
    m.pose.orientation.w = 1;
    m.pose.orientation.x = 0;
    m.pose.orientation.y = 0;
    m.pose.orientation.z = 0;
    m.color.a = 1.0;
    m.color.r = color.x();
    m.color.g = color.y();
    m.color.b = color.z();
    m.points.clear();
    geometry_msgs::Point point;
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
    m.points.push_back(point);
    point.x = position.x() + velocity.x() * scale;
    point.y = position.y() + velocity.y() * scale;
    point.z = position.z() + velocity.z() * scale;
    m.points.push_back(point);
    pub_vel.publish(m);
}


void heightCallback(const sensor_msgs::Range::ConstPtr &height_msg) {
    height = height_msg->range;
    height_time = height_msg->header.stamp.toSec();
}

void viconCallback(const nav_msgs::Odometry::ConstPtr &vicon_msg) {
    position.x() = vicon_msg->pose.pose.position.x;
    position.y() = vicon_msg->pose.pose.position.y;
    position.z() = vicon_msg->pose.pose.position.z;
    q = Eigen::Quaterniond(vicon_msg->pose.pose.orientation.w,
                           vicon_msg->pose.pose.orientation.x,
                           vicon_msg->pose.pose.orientation.y,
                           vicon_msg->pose.pose.orientation.z);
    vicon_time = vicon_msg->header.stamp.toSec();
}

void imageCallback(const sensor_msgs::Image::ConstPtr &image_msg) {
    image_time = image_msg->header.stamp.toSec();
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, image_msg->encoding);
    cv_ptr->image.copyTo(tmp);
    cv::undistort(tmp, image, cameraMatrix, distCoeffs);
    cv::imshow("optical_flow", image);
    cv::waitKey(10);
    // TODO: 1. Calculate velocity by LK Optical Flow Algorithm
    // TODO: You can use height as the z value and q(from VICON) as the orientation.
    // TODO: For this part, you can assume the UAV is flying slowly,
    // TODO: which means height changes slowly and q seldom changes.
    velocity = Eigen::Vector3d(0, 0, 0);

    // Visualize in RViz
    visualizeVelocity(position, velocity, 0, Eigen::Vector3d(1, 0, 0), pub_vel);

    velocity_gt = (position - last_position) / (vicon_time - last_vicon_time);
    visualizeVelocity(position, velocity_gt, 0, Eigen::Vector3d(0, 1, 0), pub_vel_gt);
    cout << velocity_gt << endl;
    last_position = position;
    last_vicon_time = vicon_time;

    // TODO: 2. Analyze the RMS Error here
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "opticalflow_node");
    ros::NodeHandle node;

    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    fx = cameraMatrix.at<double>(0, 0) = 362.565;
    fy = cameraMatrix.at<double>(1, 1) = 363.082;
    cx = cameraMatrix.at<double>(0, 2) = 365.486;
    cy = cameraMatrix.at<double>(1, 2) = 234.889;

    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.278765;
    distCoeffs.at<double>(1, 0) = 0.0694761;
    distCoeffs.at<double>(2, 0) = -2.86553e-05;
    distCoeffs.at<double>(3, 0) = 0.000242845;


    imageSize.height = 480;
    imageSize.width = 752;

    ros::Subscriber sub_height = node.subscribe("/tfmini_ros_node/TFmini", 10, heightCallback);
    ros::Subscriber sub_image = node.subscribe("/camera/image_raw", 10, imageCallback);
    ros::Subscriber sub_vicon = node.subscribe("/uwb_vicon_odom", 10, viconCallback);
    pub_vel = node.advertise<visualization_msgs::Marker>("/optical_flow/velocity", 1, true);
    pub_vel_gt = node.advertise<visualization_msgs::Marker>("/optical_flow/velocity_gt", 1, true);

    ros::spin();
    return 0;
}