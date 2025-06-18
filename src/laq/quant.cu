// quant.cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>    // 获取当前 CUDA Stream
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WINDOW_SIZE 32
#define MIN_FLOAT 1.17549435e-38f
#define MAX_FLOAT 3.40282347e+38f 


#include <cuda_fp16.h>
#include <iostream>
#include <math.h>

using namespace std;

__device__ int8_t rtn(float x, float scale, bool print_flag) {
    int8_t r = int8_t(roundf(x / scale));
    // if(print_flag) {
    //     printf("%f-%f-%d\n",x, scale, r);
    // }
    return r;
}


/// CUDA kernel: per-window, per-channel quantization
__global__ void quant_kernel(
    float* __restrict__ tensor,
    int B, int H, int S, int C,
    int8_t*    __restrict__ tensor_quant,
    float*     __restrict__ tensor_scale,
    int bit_width
) {
    int window_index  = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_index = threadIdx.y;

    int num_windows = S / WINDOW_SIZE;
    if (window_index >= num_windows || channel_index >= C) return;

    // find min/max
    float min_v = MAX_FLOAT;
    float max_v = MIN_FLOAT;
    int s_offset = window_index * WINDOW_SIZE;

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = s_offset; s < s_offset + WINDOW_SIZE; ++s) {
                float v = tensor[((b * H + h) * S + s) * C + channel_index];
                min_v = v < min_v ? v : min_v;
                max_v = v > max_v ? v : max_v;
            }
        }
    }

    // calculate scale：h_scale = (max_v - min_v) / ((2^bit_width)-1)
    float range   = max_v - min_v;
    float denom   = (1u << bit_width) - 1u;
    float h_scale = range / denom;
    tensor_scale[window_index * C + channel_index] = h_scale;
    // quantization
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = s_offset; s < s_offset + WINDOW_SIZE; ++s) {
                float v   = tensor[((b * H + h) * S + s) * C + channel_index];
                bool f = false;
                if(b == 0 && h == 0 && s == s_offset)f = true;
                tensor_quant[((b * H + h) * S + s) * C + channel_index] = rtn(v, h_scale, f);
            }
        }
    }
}


__global__ void dequant_kernel(
    float* __restrict__ tensor, // output
    int B, int H, int S, int C,
    int8_t*    __restrict__ tensor_quant, // input
    float*     __restrict__ tensor_scale, // scale factors
    int bit_width
) {
    int window_index  = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_index = threadIdx.y;

    int num_windows = S / WINDOW_SIZE;
    if (window_index >= num_windows || channel_index >= C) return;

    float scale = tensor_scale[window_index * C + channel_index];

    // quantization
    int s_offset = window_index * WINDOW_SIZE;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int s = s_offset; s < s_offset + WINDOW_SIZE; ++s) {
                int8_t v   = tensor_quant[((b * H + h) * S + s) * C + channel_index];
                float d_v = float(v) * scale;
                tensor[((b * H + h) * S + s) * C + channel_index] = d_v;
            }
        }
    }
}


// pybind launcher
void quant_launcher(
    at::Tensor a,       // float16
    at::Tensor out_q,   // int8
    at::Tensor out_s,   // float
    int B, int H, int S, int C,
    int bit_width
) {
    TORCH_CHECK(a.is_cuda() && out_q.is_cuda() && out_s.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "Input must be Half");
    TORCH_CHECK(out_q.scalar_type() == at::kChar, "out_q must be int8");
    TORCH_CHECK(out_s.scalar_type() == at::kFloat, "out_s must be float");

    int num_windows = S / WINDOW_SIZE;
    int threads_per_block = 256;
    int threads_y = C;
    int threads_x = threads_per_block / threads_y;
    threads_x = min(threads_x, num_windows);
    int blocks = (num_windows + threads_x - 1) / threads_x;

    dim3 block_dim(threads_x, threads_y);
    dim3 grid_dim(blocks, 1);

    // current cuda stream
    c10::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = cuda_stream.stream();

    float* tensor_ptr = a.data_ptr<float>();

    int8_t*      q_ptr      = out_q.data_ptr<int8_t>();
    float*       s_ptr      = out_s.data_ptr<float>();

    quant_kernel<<<grid_dim, block_dim, 0, stream>>>(
        tensor_ptr, B, H, S, C,
        q_ptr, s_ptr, bit_width
    );

    // check kernel error
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "quant_kernel launch failed: ", cudaGetErrorString(err));
}

// pybind launcher
void dequant_launcher(
    at::Tensor a,       // float16
    at::Tensor out_q,   // int8
    at::Tensor out_s,   // float
    int B, int H, int S, int C,
    int bit_width
) {
    TORCH_CHECK(a.is_cuda() && out_q.is_cuda() && out_s.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "Input must be Half");
    TORCH_CHECK(out_q.scalar_type() == at::kChar, "out_q must be int8");
    TORCH_CHECK(out_s.scalar_type() == at::kFloat, "out_s must be float");

    int num_windows = S / WINDOW_SIZE;
    int threads_per_block = 256;
    int threads_y = C;
    int threads_x = threads_per_block / threads_y;
    threads_x = min(threads_x, num_windows);
    int blocks = (num_windows + threads_x - 1) / threads_x;

    dim3 block_dim(threads_x, threads_y);
    dim3 grid_dim(blocks, 1);

    // current cuda stream
    c10::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t stream = cuda_stream.stream();

    float* tensor_ptr = a.data_ptr<float>();
    int8_t*      q_ptr      = out_q.data_ptr<int8_t>();
    float* s_ptr = out_s.data_ptr<float>();

    dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(
        tensor_ptr, B, H, S, C,
        q_ptr, s_ptr, bit_width
    );

    // check kernel error
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "quant_kernel launch failed: ", cudaGetErrorString(err));
}

/// pybind111 module name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant",
          &quant_launcher,
          "Per-window per-channel quantization (CUDA)",
          py::arg("input"),
          py::arg("out_q"),
          py::arg("out_scale"),
          py::arg("B"),
          py::arg("H"),
          py::arg("S"),
          py::arg("C"),
          py::arg("bit_width"));
    m.def("dequant",
          &dequant_launcher,
          "Per-window per-channel quantization (CUDA)",
          py::arg("input"),
          py::arg("out_q"),
          py::arg("out_scale"),
          py::arg("B"),
          py::arg("H"),
          py::arg("S"),
          py::arg("C"),
          py::arg("bit_width"));
}