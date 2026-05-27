#pragma once
#include <opencv2/opencv.hpp>

namespace YOLO {
    class Camera {
    public:
        Camera(const std::string& device = "/dev/video0", int width = 640, int height = 480, int fps = 30);
        ~Camera();

        bool isOpened() const;
        bool read(cv::Mat& frame);
        void release();

        bool setResolution(int width, int height);
        bool setFPS(int fps);
        bool setFrameFormat(const std::string& fourcc);

        int width() const { return width_; }
        int height() const { return height_; }

    private:
        cv::VideoCapture cap_;
        int width_;
        int height_;
        int fps_;
        bool use_gstreamer_;
    };
}
