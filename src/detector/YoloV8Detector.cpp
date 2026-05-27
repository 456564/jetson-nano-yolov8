#include "YoloV8Detector.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <algorithm>

void launch_preprocess_letterbox(
    const uint8_t* d_src, int src_w, int src_h, int src_step,
    float* d_dst, int dst_w, int dst_h,
    float scale, int pad_left, int pad_top,
    int pad_right, int pad_bottom,
    cudaStream_t stream);

namespace YOLO {

    YoloV8Detector::YoloV8Detector(const std::string& engine_path, int input_size,
                                     int cam_width, int cam_height)
        : input_size_(input_size), cam_width_(cam_width), cam_height_(cam_height) {

        loadEngine(engine_path);

        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) throw std::runtime_error("创建 TRT Runtime 失败");

        engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
        if (!engine_) throw std::runtime_error("反序列化 Engine 失败");
        delete runtime_;
        runtime_ = nullptr;

        context_ = engine_->createExecutionContext();
        if (!context_) throw std::runtime_error("创建执行上下文失败");

        input_index_  = engine_->getBindingIndex("images");
        output_index_ = engine_->getBindingIndex("output0");

        input_size_bytes_ = 1 * 3 * input_size_ * input_size_ * sizeof(float);
        auto out_dims = engine_->getBindingDimensions(output_index_);
        output_size_bytes_ = 1 * out_dims.d[1] * out_dims.d[2] * sizeof(float);

        frame_bytes_ = cam_width_ * cam_height_ * 3 * sizeof(uint8_t);

        cudaMallocHost(&cpu_input_buffer_,  frame_bytes_);
        cudaMallocHost(&cpu_output_buffer_, output_size_bytes_);

        cudaMalloc(&gpu_buffers_[input_index_],  input_size_bytes_);
        cudaMalloc(&gpu_input_uint8_,             frame_bytes_);
        cudaMalloc(&gpu_buffers_[output_index_], output_size_bytes_);

        cudaStreamCreate(&stream_);
    }

    YoloV8Detector::~YoloV8Detector() {
        cudaStreamDestroy(stream_);
        cudaFree(gpu_buffers_[input_index_]);
        cudaFree(gpu_input_uint8_);
        cudaFree(gpu_buffers_[output_index_]);
        cudaFreeHost(cpu_input_buffer_);
        cudaFreeHost(cpu_output_buffer_);
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_)  { delete engine_;  engine_  = nullptr; }
    }

    void YoloV8Detector::loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) throw std::runtime_error("找不到 engine 文件: " + engine_path);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        engine_data_.resize(size);
        file.read(engine_data_.data(), size);
    }

    void YoloV8Detector::computeLetterboxParams(int orig_w, int orig_h,
                                                 float& scale, int& pad_w, int& pad_h) {
        scale = std::min((float)input_size_ / orig_w, (float)input_size_ / orig_h);
        int new_w = (int)(orig_w * scale);
        int new_h = (int)(orig_h * scale);
        pad_w = input_size_ - new_w;
        pad_h = input_size_ - new_h;
    }

    const float* YoloV8Detector::infer(const cv::Mat& img,
                                        float& scale, int& pad_w, int& pad_h) {
        int orig_w = img.cols, orig_h = img.rows;
        computeLetterboxParams(orig_w, orig_h, scale, pad_w, pad_h);

        int new_w = (int)(orig_w * scale);
        int new_h = (int)(orig_h * scale);
        int pad_left = pad_w / 2;
        int pad_top  = pad_h / 2;
        int pad_right  = pad_w - pad_left;
        int pad_bottom = pad_h - pad_top;

        size_t actual_frame = orig_w * orig_h * 3;
        memcpy(cpu_input_buffer_, img.data, actual_frame);
        cudaMemcpyAsync(gpu_input_uint8_, cpu_input_buffer_, actual_frame,
                        cudaMemcpyHostToDevice, stream_);

        launch_preprocess_letterbox(
            (const uint8_t*)gpu_input_uint8_, orig_w, orig_h, (int)img.step,
            (float*)gpu_buffers_[input_index_], input_size_, input_size_,
            scale, pad_left, pad_top, pad_right, pad_bottom, stream_);

        context_->enqueueV2(gpu_buffers_, stream_, nullptr);

        cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[output_index_],
                        output_size_bytes_, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        return (const float*)cpu_output_buffer_;
    }
}
