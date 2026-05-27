#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "detector/YoloV8Detector.h"
#include "detector/YoloV8PostProcess.h"
#include "utils/Types.h"
#include "camera/Camera.h"

static void get_color(int id, cv::Scalar& color) {
    static cv::RNG rng(0xFF);
    color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

int main() {
    try {
        const int CAM_W = 640, CAM_H = 480, INPUT_SIZE = 640;

        std::cout << "[1] 初始化 TensorRT..." << std::endl;
        YOLO::YoloV8Detector detector("../models/yolov8n_fp16.engine", INPUT_SIZE, CAM_W, CAM_H);

        std::cout << "[2] 初始化摄像头..." << std::endl;
        YOLO::Camera camera("/dev/video0", CAM_W, CAM_H, 30);
        if (!camera.isOpened()) {
            std::cerr << "无法打开摄像头" << std::endl;
            return -1;
        }

        std::cout << "[3] 开始全链路性能剖析 (按 'q' 退出)..." << std::endl;

        cv::Mat frame;
        double total_cap = 0, total_infer = 0, total_post = 0, total_draw = 0;
        int frame_count = 0;

        // 预热
        for (int i = 0; i < 5; i++) {
            camera.read(frame);
            float scale; int pad_w, pad_h;
            detector.infer(frame, scale, pad_w, pad_h);
        }

        while (true) {
            auto t1 = std::chrono::high_resolution_clock::now();

            if (!camera.read(frame)) break;
            auto t2 = std::chrono::high_resolution_clock::now();

            float scale; int pad_w, pad_h;
            const float* output = detector.infer(frame, scale, pad_w, pad_h);
            auto t3 = std::chrono::high_resolution_clock::now();

            std::vector<YOLO::Object> objects;
            YOLO::decodeOutputs(output, 80, 8400, objects,
                                scale, pad_w, pad_h,
                                frame.cols, frame.rows, 0.25f);
            YOLO::nms(objects, 0.45f);
            auto t4 = std::chrono::high_resolution_clock::now();

            for (const auto& obj : objects) {
                cv::Scalar color;
                get_color(obj.label, color);
                cv::rectangle(frame, cv::Point(obj.x, obj.y),
                              cv::Point(obj.x + obj.w, obj.y + obj.h), color, 2);
                char text[256];
                sprintf(text, "%s %.1f%%", YOLO::CLASS_NAMES[obj.label], obj.prob * 100);
                cv::putText(frame, text, cv::Point(obj.x, obj.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
            auto t5 = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> cap_dur   = t2 - t1;
            std::chrono::duration<double, std::milli> infer_dur = t3 - t2;
            std::chrono::duration<double, std::milli> post_dur  = t4 - t3;
            std::chrono::duration<double, std::milli> draw_dur  = t5 - t4;

            total_cap   += cap_dur.count();
            total_infer += infer_dur.count();
            total_post  += post_dur.count();
            total_draw  += draw_dur.count();
            frame_count++;

            if (frame_count % 30 == 0) {
                float avg_total = (total_cap + total_infer + total_post + total_draw) / 30.0f;
                float fps = 1000.0f / avg_total;

                printf("\n=== [性能剖析] 平均30帧耗时 ===\n");
                printf("1. 图像采集:     %5.2f ms\n", total_cap   / 30);
                printf("2. 预处理+推理:  %5.2f ms  <-- TensorRT\n", total_infer / 30);
                printf("3. 后处理:       %5.2f ms\n", total_post  / 30);
                printf("4. 画框显示:     %5.2f ms\n", total_draw  / 30);
                printf("============================\n");
                printf(">>> 实际吞吐 FPS: %.2f <<<\n\n", fps);

                total_cap = 0; total_infer = 0; total_post = 0; total_draw = 0;
            }

            std::chrono::duration<double, std::milli> total = t5 - t1;
            char fps_text[32];
            sprintf(fps_text, "%.1f ms", total.count());
            cv::putText(frame, fps_text, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            cv::imshow("YOLOv8 Profiling", frame);
            if (cv::waitKey(1) == 'q') break;
        }

    } catch (const std::exception& e) {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
