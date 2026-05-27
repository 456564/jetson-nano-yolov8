#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace YOLO {
    class YoloV8Detector {
    public:
        YoloV8Detector(const std::string& engine_path, int input_size,
                       int cam_width, int cam_height);
        ~YoloV8Detector();

        const float* infer(const cv::Mat& img, float& scale, int& pad_w, int& pad_h);
        const float* outputBuffer() const { return (const float*)cpu_output_buffer_; }

    private:
        void loadEngine(const std::string& engine_path);
        void computeLetterboxParams(int orig_w, int orig_h,
                                    float& scale, int& pad_w, int& pad_h);

        nvinfer1::IRuntime* runtime_ = nullptr;
        nvinfer1::ICudaEngine* engine_ = nullptr;
        nvinfer1::IExecutionContext* context_ = nullptr;
        cudaStream_t stream_;

        void* gpu_buffers_[2];
        void* gpu_input_uint8_ = nullptr;
        void* cpu_input_buffer_ = nullptr;
        void* cpu_output_buffer_ = nullptr;
        int input_index_;
        int output_index_;
        size_t input_size_bytes_;
        size_t output_size_bytes_;
        size_t frame_bytes_;

        int input_size_;
        int cam_width_;
        int cam_height_;
        std::vector<char> engine_data_;

        class Logger : public nvinfer1::ILogger {
            void log(Severity severity, const char* msg) noexcept override {
                if (severity <= Severity::kWARNING)
                    std::cout << "[TRT] " << msg << std::endl;
            }
        };
        Logger logger_;
    };
}
