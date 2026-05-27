#include "Camera.h"
#include <iostream>

namespace YOLO {

    Camera::Camera(const std::string& device, int width, int height, int fps)
        : width_(width), height_(height), fps_(fps), use_gstreamer_(false) {

        std::string gst_pipeline =
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) +
            ", height=(int)" + std::to_string(height) +
            ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) + "/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1";

        cap_.open(gst_pipeline, cv::CAP_GSTREAMER);
        if (cap_.isOpened()) {
            cv::Mat test;
            if (cap_.read(test) && !test.empty()) {
                use_gstreamer_ = true;
                std::cout << "[Camera] GStreamer nvarguscamerasrc 打开成功" << std::endl;
                return;
            }
            cap_.release();
            std::cout << "[Camera] GStreamer 无画面，回退到 V4L2: " << device << std::endl;
        } else {
            std::cout << "[Camera] GStreamer 失败，回退到 V4L2: " << device << std::endl;
        }
        cap_.open(device, cv::CAP_V4L2);
        if (cap_.isOpened()) {
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            cap_.set(cv::CAP_PROP_FPS, fps);
        }
    }

    Camera::~Camera() {
        release();
    }

    bool Camera::isOpened() const {
        return cap_.isOpened();
    }

    bool Camera::read(cv::Mat& frame) {
        return cap_.read(frame);
    }

    void Camera::release() {
        if (cap_.isOpened()) cap_.release();
    }

    bool Camera::setResolution(int width, int height) {
        width_ = width;
        height_ = height;
        return cap_.set(cv::CAP_PROP_FRAME_WIDTH, width) &&
               cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }

    bool Camera::setFPS(int fps) {
        fps_ = fps;
        return cap_.set(cv::CAP_PROP_FPS, fps);
    }

    bool Camera::setFrameFormat(const std::string& fourcc) {
        if (fourcc.size() != 4) return false;
        int code = cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
        return cap_.set(cv::CAP_PROP_FOURCC, code);
    }

}
