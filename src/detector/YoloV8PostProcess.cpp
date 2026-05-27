#include "YoloV8PostProcess.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace YOLO {

    void nms(std::vector<Object>& objects, float nms_threshold) {
        std::sort(objects.begin(), objects.end(),
                  [](const Object& a, const Object& b) { return a.prob > b.prob; });

        int n = (int)objects.size();
        for (int i = 0; i < n; ++i) {
            if (objects[i].prob == 0) continue;
            float ix1 = objects[i].x, iy1 = objects[i].y;
            float ix2 = objects[i].x + objects[i].w, iy2 = objects[i].y + objects[i].h;
            float iarea = (ix2 - ix1) * (iy2 - iy1);

            for (int j = i + 1; j < n; ++j) {
                if (objects[j].prob == 0) continue;
                float jx1 = objects[j].x, jy1 = objects[j].y;
                float jx2 = objects[j].x + objects[j].w, jy2 = objects[j].y + objects[j].h;

                float inter_w = std::max(0.f, std::min(ix2, jx2) - std::max(ix1, jx1));
                float inter_h = std::max(0.f, std::min(iy2, jy2) - std::max(iy1, jy1));
                float inter = inter_w * inter_h;
                float jarea = (jx2 - jx1) * (jy2 - jy1);
                float iou = inter / (iarea + jarea - inter);

                if (iou > nms_threshold) objects[j].prob = 0;
            }
        }
        objects.erase(std::remove_if(objects.begin(), objects.end(),
                       [](const Object& o) { return o.prob == 0; }), objects.end());
    }

    void decodeOutputs(const float* output, int num_classes, int num_boxes,
                       std::vector<Object>& objects,
                       float scale, int pad_w, int pad_h,
                       int orig_w, int orig_h, float conf_thresh) {

        std::vector<float> max_scores(num_boxes, -1.f);
        std::vector<int>   class_ids(num_boxes, -1);

#pragma omp parallel for
        for (int i = 0; i < num_boxes; i++) {
            float best = -1.f;
            int   cls  = -1;
            for (int j = 0; j < num_classes; j++) {
                float s = output[(4 + j) * num_boxes + i];
                if (s > best) { best = s; cls = j; }
            }
            max_scores[i] = best;
            class_ids[i]  = cls;
        }

        float pad_x = pad_w * 0.5f;
        float pad_y = pad_h * 0.5f;
        float inv_scale = 1.0f / scale;

        objects.reserve(128);
        for (int i = 0; i < num_boxes; i++) {
            if (max_scores[i] < conf_thresh) continue;

            float cx = output[0 * num_boxes + i];
            float cy = output[1 * num_boxes + i];
            float bw = output[2 * num_boxes + i];
            float bh = output[3 * num_boxes + i];

            float x1 = (cx - bw * 0.5f - pad_x) * inv_scale;
            float y1 = (cy - bh * 0.5f - pad_y) * inv_scale;
            float x2 = (cx + bw * 0.5f - pad_x) * inv_scale;
            float y2 = (cy + bh * 0.5f - pad_y) * inv_scale;

            x1 = std::max(0.f, std::min(x1, (float)orig_w));
            y1 = std::max(0.f, std::min(y1, (float)orig_h));
            x2 = std::max(0.f, std::min(x2, (float)orig_w));
            y2 = std::max(0.f, std::min(y2, (float)orig_h));

            objects.push_back({x1, y1, x2 - x1, y2 - y1, class_ids[i], max_scores[i]});
        }
    }
}
