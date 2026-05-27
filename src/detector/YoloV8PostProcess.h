#pragma once
#include <vector>
#include "utils/Types.h"

namespace YOLO {
    void nms(std::vector<Object>& objects, float nms_threshold);
    
    void decodeOutputs(const float* output, int num_classes, int num_boxes,
                       std::vector<Object>& objects,
                       float scale, int pad_w, int pad_h,
                       int orig_w, int orig_h, float conf_thresh);
}
