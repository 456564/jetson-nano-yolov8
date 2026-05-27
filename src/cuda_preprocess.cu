#include <cstdint>
#include <cuda_runtime.h>

#define THREADS_X 16
#define THREADS_Y 16

__global__ void preprocess_letterbox_kernel(
    const uint8_t* src, int src_w, int src_h, int src_step,
    float* dst, int dst_w, int dst_h,
    float scale, int pad_left, int pad_top,
    int pad_right, int pad_bottom)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= dst_w || oy >= dst_h) return;

    float r, g, b;

    if (ox < pad_left || ox >= dst_w - pad_right ||
        oy < pad_top  || oy >= dst_h - pad_bottom) {
        r = g = b = 114.0f / 255.0f;
    } else {
        int sx = (int)((ox - pad_left) / scale + 0.5f);
        int sy = (int)((oy - pad_top)  / scale + 0.5f);
        sx = max(0, min(sx, src_w - 1));
        sy = max(0, min(sy, src_h - 1));

        int idx = sy * src_step + sx * 3;
        b = src[idx + 0] / 255.0f;  // BGR -> RGB reorder
        g = src[idx + 1] / 255.0f;
        r = src[idx + 2] / 255.0f;
    }

    int plane = dst_h * dst_w;
    dst[0 * plane + oy * dst_w + ox] = r;
    dst[1 * plane + oy * dst_w + ox] = g;
    dst[2 * plane + oy * dst_w + ox] = b;
}

void launch_preprocess_letterbox(
    const uint8_t* d_src, int src_w, int src_h, int src_step,
    float* d_dst, int dst_w, int dst_h,
    float scale, int pad_left, int pad_top,
    int pad_right, int pad_bottom,
    cudaStream_t stream)
{
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((dst_w + THREADS_X - 1) / THREADS_X,
              (dst_h + THREADS_Y - 1) / THREADS_Y);
    preprocess_letterbox_kernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, src_step,
        d_dst, dst_w, dst_h,
        scale, pad_left, pad_top, pad_right, pad_bottom);
}
