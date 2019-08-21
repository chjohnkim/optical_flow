#include <ros/ros.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include "visualization_msgs/Marker.h"

#include <algorithm>
#include <functional>

using namespace std;
using namespace cv;

double height, last_height;
double vicon_time, height_time, last_height_time, last_vicon_time, image_time, last_image_time;
Eigen::Vector3d velocity, position, last_position, velocity_gt;
Eigen::Quaterniond q;
ros::Publisher pub_vel, pub_vel_gt;
cv::Mat tmp, image, img0, img1;
double fx, fy, cx, cy;
cv::Mat cameraMatrix, distCoeffs;
cv::Size imageSize;

std::vector<cv::Point2f> p1, p0, p0r, p0new;
double vx, vy, vz, vx_filtered, vy_filtered, vz_filtered, pixel_vx, pixel_vy;
double sum_x=0, sum_y=0, sum_x_filtered, sum_y_filtered;

void visualizeVelocity(Eigen::Vector3d position, Eigen::Vector3d velocity,
                       int id, Eigen::Vector3d color, ros::Publisher pub_vel) {
    double scale = 10;
    visualization_msgs::Marker m;
    m.header.frame_id = "base_link";
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

    //cv::imshow("optical_flow", image);
    cv::waitKey(10);
    // TODO: 1. Calculate velocity by LK Optical Flow Algorithm
    // TODO: You can use height as the z value and q(from VICON) as the orientation.
    // TODO: For this part, you can assume the UAV is flying slowly,
    // TODO: which means height changes slowly and q seldom changes.
    
    
    //std::cout << "This quaternion consists of a scalar " << q.w() << " and a vector " << std::endl << q.vec() << std::endl;
    Size subPixWinSize(10,10), winSize(31,31);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

    const int MAX_COUNT = 15;
    vector<uchar> status, status_tmp;
    vector<float> err;
    std::vector<cv::Point2f> p1_filtered, p0_filtered;

    image.copyTo(img1);
    size_t i, k;
    //If first loop, initialize
    if(img0.empty())
    {   cout << "Initializing..." << endl; 
        img1.copyTo(img0);
        // 480 by 752
        for(i=1; i < 480/(752/MAX_COUNT); i++)
        {   for(k=1; k < 752/(752/MAX_COUNT); k++)
            {   p0.push_back(Point2f(k*(752/MAX_COUNT), i*(752/MAX_COUNT)));
            }
        }
    }    

    //If less than 100 corners, add more corners to track
    //if (p0.size() < MAX_COUNT)
    //{
    //    cv::goodFeaturesToTrack(img0, p0new, 50, 0.01, 10, Mat(), 3.0, 0.0, 0.04);
    //    cv::cornerSubPix(img0, p0new, subPixWinSize, Size(-1,-1), termcrit);
    //    p0.insert(p0.end(), p0new.begin(), p0new.end());
    //}
    p0.clear();
    for(i=1; i < 480/(752/MAX_COUNT); i++)
    {   for(k=1; k < 752/(752/MAX_COUNT); k++)
        {   p0.push_back(Point2f(k*(752/MAX_COUNT), i*(752/MAX_COUNT)));
        }
    }
    //Optical Flow (LK)
    cv::calcOpticalFlowPyrLK(img0, img1, p0, p1, status, err, winSize, 3, termcrit, 0, 0.001);
    
    //Filter corners that move by more than 1 pixel to satisfy small motion assumption
    cv::calcOpticalFlowPyrLK(img1, img0, p1, p0r, status_tmp, err, winSize, 3, termcrit, 0, 0.001);
    std::vector<cv::Point2f> d(p0.size());
    
    //for(i=0; i < p0.size(); i++)
    //{
    //    d[i] = Point2f(p0[i].x - p0r[i].x, p0[i].y - p0r[i].y);
    //    if(d[i].x < 1 && d[i].x > -1 && d[i].y < 1 && d[i].y > -1 && status[i])
    //    {
    //        p1_filtered.push_back(Point2f(p1[i].x, p1[i].y));
    //        p0_filtered.push_back(Point2f(p0[i].x, p0[i].y));
    //    }
    //}
    vector<uchar> mask;
    cv::findFundamentalMat(p1, p0, FM_RANSAC, 3, 0.99, mask);
    for(i=0; i < p0.size(); i++)
    {   d[i] = Point2f(p0[i].x - p0r[i].x, p0[i].y - p0r[i].y);
        if(status_tmp[i] && status[i] && mask[i] && d[i].x < 1 && d[i].x > -1 && d[i].y < 1 && d[i].y > -1)
        //if(mask[i] && status[i] && d[i].x < 1 && d[i].x > -1 && d[i].y < 1 && d[i].y > -1)
        {
            p1_filtered.push_back(Point2f(p1[i].x, p1[i].y));
            p0_filtered.push_back(Point2f(p0[i].x, p0[i].y));
        }
    }
    cout << p1_filtered.size() << endl;


    //Calculate the average differnce from pixel movement
    sum_x = sum_y = 0;
    std::vector<cv::Point2f> d_filtered(p0_filtered.size());
    for(i=0; i < p0_filtered.size(); i++)
    {
        sum_x = sum_x + (p1_filtered[i].x - p0_filtered[i].x);
        sum_y = sum_y + (p1_filtered[i].y - p0_filtered[i].y);
    }
    sum_x = sum_x / p1_filtered.size();
    sum_y = sum_y / p1_filtered.size();
    
    cv::Point2f center, sum;
    center.x = 376;
    center.y = 240;
    sum.x = 376+sum_x;
    sum.y = 240+sum_y;
    //Visualization
    for(i=0; i < p1_filtered.size(); i++)
    {
        //circle( image, p1_filtered[i], 3, Scalar(255,255,255), -1, 8);
        //line( image, p1_filtered[i], p0_filtered[i], Scalar(255,255,255), 1, 8, 0);
        line( image, sum, center, Scalar(255,255,255), 10, 8, 0);
        cv::imshow("optical_flow", image);
        
    }

    std::swap(p1_filtered, p0);
    cv::swap(img0, img1);
    

    //z-velocity from height information
    //vz = (height - last_height) / (height_time - last_height_time);
    //#vz_filtered += 0.25 * (vz - vz_filtered);
    
    
    //pixel velocity
    pixel_vx = sum_x/(image_time - last_image_time);
    pixel_vy = sum_y/(image_time - last_image_time);
    
    velocity_gt = (position - last_position) / (vicon_time - last_vicon_time);

    //camera x and y velocity 
    vx = (pixel_vx - (cx/height)*(0))/(-fx/height);
    vy = (pixel_vy - (cy/height)*(0))/(-fy/height);
    
    vx_filtered += 0.4*(vx - vx_filtered);
    vy_filtered += 0.4*(vy - vy_filtered);

    last_height = height;
    last_height_time = height_time;
    last_image_time = image_time;
    // My code ends here
    
    velocity = Eigen::Vector3d(vy_filtered, vx_filtered, velocity_gt[2]);


    // Visualize in RViz
    visualizeVelocity(position, velocity, 0, Eigen::Vector3d(1, 0, 0), pub_vel);

    visualizeVelocity(position, velocity_gt, 0, Eigen::Vector3d(0, 1, 0), pub_vel_gt);

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


    ros::Subscriber sub_height = node.subscribe("/tfmini_ros_node/TFmini", 10, heightCallback);
    ros::Subscriber sub_image = node.subscribe("/camera/image_raw", 10, imageCallback);
    ros::Subscriber sub_vicon = node.subscribe("/uwb_vicon_odom", 10, viconCallback);
    pub_vel = node.advertise<visualization_msgs::Marker>("/optical_flow/velocity", 1, true);
    pub_vel_gt = node.advertise<visualization_msgs::Marker>("/optical_flow/velocity_gt", 1, true);

    ros::spin();
    return 0;
}