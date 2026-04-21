#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr float kScoreClamp = 8.0f;
constexpr int kFilterLen = 8;
constexpr int kBlockSize1D = 256;

__device__ __constant__ float kDb4Low[kFilterLen] = {
    -0.010597401785069032f,
    0.0328830116668852f,
    0.030841381835560764f,
    -0.18703481171888114f,
    -0.027983769416859854f,
    0.6308807679298589f,
    0.7148465705529157f,
    0.2303778133088965f,
};

__device__ __constant__ float kDb4High[kFilterLen] = {
    -0.2303778133088965f,
    0.7148465705529157f,
    -0.6308807679298589f,
    -0.027983769416859854f,
    0.18703481171888114f,
    0.030841381835560764f,
    -0.0328830116668852f,
    -0.010597401785069032f,
};

#if __CUDA_ARCH__ < 600
__device__ double atomic_add_double(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ __forceinline__ double atomic_add_double(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

__device__ __forceinline__ int reflect_index(int idx, int len) {
    if (len <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= len) {
        if (idx < 0) {
            idx = -idx - 1;
        }
        if (idx >= len) {
            idx = 2 * len - idx - 1;
        }
    }
    return idx;
}

__device__ __forceinline__ float band_response(float lh, float hl, float hh) {
    return fmaxf(fabsf(lh), fmaxf(fabsf(hl), 0.5f * fabsf(hh)));
}

__global__ void pad_reflect_and_mask_kernel(
    const uint8_t* image,
    const uint8_t* circle_mask,
    int width,
    int height,
    int padded_width,
    int padded_height,
    float* output
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padded_width || y >= padded_height) {
        return;
    }

    const int src_x = reflect_index(x, width);
    const int src_y = reflect_index(y, height);
    const int src_idx = src_y * width + src_x;
    output[y * padded_width + x] = circle_mask[src_idx] ? (image[src_idx] * (1.0f / 255.0f)) : 0.0f;
}

__global__ void convolve_rows_kernel(
    const float* src,
    float* low_rows,
    float* high_rows,
    int width,
    int height,
    int stride
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    constexpr int kAnchor = kFilterLen / 2;
    float sum_low = 0.0f;
    float sum_high = 0.0f;
    const int row = y * width;
    for (int k = 0; k < kFilterLen; ++k) {
        const int sx = reflect_index(x + (k - kAnchor) * stride, width);
        const float value = src[row + sx];
        sum_low += kDb4Low[k] * value;
        sum_high += kDb4High[k] * value;
    }
    low_rows[row + x] = sum_low;
    high_rows[row + x] = sum_high;
}

__global__ void convolve_cols_kernel(
    const float* low_rows,
    const float* high_rows,
    float* approx_out,
    float* lh,
    float* hl,
    float* hh,
    int width,
    int height,
    int stride
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    constexpr int kAnchor = kFilterLen / 2;
    float sum_ll = 0.0f;
    float sum_lh = 0.0f;
    float sum_hl = 0.0f;
    float sum_hh = 0.0f;
    for (int k = 0; k < kFilterLen; ++k) {
        const int sy = reflect_index(y + (k - kAnchor) * stride, height);
        const int idx = sy * width + x;
        const float low = low_rows[idx];
        const float high = high_rows[idx];
        sum_ll += kDb4Low[k] * low;
        sum_lh += kDb4High[k] * low;
        sum_hl += kDb4Low[k] * high;
        sum_hh += kDb4High[k] * high;
    }
    const int out_idx = y * width + x;
    approx_out[out_idx] = sum_ll;
    lh[out_idx] = sum_lh;
    hl[out_idx] = sum_hl;
    hh[out_idx] = sum_hh;
}

__global__ void reduce_response_stats_kernel(
    const float* lh,
    const float* hl,
    const float* hh,
    const uint8_t* image,
    const uint8_t* circle_mask,
    int width,
    int height,
    int padded_width,
    double* sum_out,
    double* sq_sum_out,
    unsigned long long* count_out
) {
    __shared__ double sh_sum[kBlockSize1D];
    __shared__ double sh_sq_sum[kBlockSize1D];
    __shared__ unsigned int sh_count[kBlockSize1D];

    const int tid = threadIdx.x;
    const int total = width * height;
    double local_sum = 0.0;
    double local_sq_sum = 0.0;
    unsigned int local_count = 0;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < total; idx += blockDim.x * gridDim.x) {
        if (circle_mask[idx] == 0 || image[idx] == 0) {
            continue;
        }
        const int y = idx / width;
        const int x = idx - y * width;
        const int padded_idx = y * padded_width + x;
        const float response = band_response(lh[padded_idx], hl[padded_idx], hh[padded_idx]);
        local_sum += static_cast<double>(response);
        local_sq_sum += static_cast<double>(response) * static_cast<double>(response);
        local_count += 1U;
    }

    sh_sum[tid] = local_sum;
    sh_sq_sum[tid] = local_sq_sum;
    sh_count[tid] = local_count;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sq_sum[tid] += sh_sq_sum[tid + offset];
            sh_count[tid] += sh_count[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_double(sum_out, sh_sum[0]);
        atomic_add_double(sq_sum_out, sh_sq_sum[0]);
        atomicAdd(count_out, static_cast<unsigned long long>(sh_count[0]));
    }
}

__global__ void compute_threshold_kernel(
    const double* sum,
    const double* sq_sum,
    const unsigned long long* count,
    float sigma,
    float* threshold_out
) {
    const unsigned long long count_value = *count;
    if (count_value == 0ULL) {
        *threshold_out = 0.0f;
        return;
    }
    const double mean = *sum / static_cast<double>(count_value);
    const double variance = fmax((*sq_sum / static_cast<double>(count_value)) - mean * mean, 0.0);
    *threshold_out = static_cast<float>(mean + static_cast<double>(sigma) * sqrt(variance));
}

__global__ void update_score_accum_kernel(
    const float* lh,
    const float* hl,
    const float* hh,
    const uint8_t* image,
    const uint8_t* circle_mask,
    int width,
    int height,
    int padded_width,
    const float* threshold,
    float* score_accum
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }

    const float thr = *threshold;
    if (thr <= 0.0f || circle_mask[idx] == 0 || image[idx] == 0) {
        return;
    }

    const int y = idx / width;
    const int x = idx - y * width;
    const int padded_idx = y * padded_width + x;
    const float response = band_response(lh[padded_idx], hl[padded_idx], hh[padded_idx]);
    const float ratio = response / thr;
    if (ratio > score_accum[idx]) {
        score_accum[idx] = ratio;
    }
}

__global__ void finalize_outputs_kernel(
    const float* score_accum,
    const uint8_t* image,
    const uint8_t* circle_mask,
    int total,
    uint8_t* binary_out,
    uint8_t* score_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    if (circle_mask[idx] == 0 || image[idx] == 0) {
        binary_out[idx] = 0;
        score_out[idx] = 0;
        return;
    }

    const float value = score_accum[idx];
    binary_out[idx] = (value > 1.0f) ? 1 : 0;
    const float scaled = fminf(value, kScoreClamp) * (255.0f / kScoreClamp);
    score_out[idx] = static_cast<uint8_t>(lrintf(scaled));
}

inline int check_cuda(cudaError_t status) {
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

inline int check_launch() {
    return check_cuda(cudaPeekAtLastError());
}

}  // namespace

extern "C" int wavelet_cuda_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return 0;
    }
    return count;
}

namespace {

int process_one_image_u8(
    const uint8_t* image_ptr,
    const uint8_t* device_circle_mask,
    int width,
    int height,
    int level,
    float threshold_sigma,
    uint8_t* device_image,
    uint8_t* device_binary,
    uint8_t* device_score,
    float* device_approx_a,
    float* device_approx_b,
    float* device_low_rows,
    float* device_high_rows,
    float* device_lh,
    float* device_hl,
    float* device_hh,
    float* device_score_accum,
    double* device_sum,
    double* device_sq_sum,
    unsigned long long* device_count,
    float* device_threshold,
    uint8_t* binary_out,
    uint8_t* score_out
) {
    const int active_level = level > 0 ? level : 1;
    const int scale = 1 << active_level;
    const int padded_width = ((width + scale - 1) / scale) * scale;
    const int padded_height = ((height + scale - 1) / scale) * scale;
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t total_bytes = total * sizeof(uint8_t);
    const dim3 block2d(16, 16);
    const dim3 grid2d(
        (padded_width + block2d.x - 1) / block2d.x,
        (padded_height + block2d.y - 1) / block2d.y
    );
    const int grid1d = std::max(1, std::min((width * height + kBlockSize1D - 1) / kBlockSize1D, 4096));

    int status = check_cuda(cudaMemcpy(device_image, image_ptr, total_bytes, cudaMemcpyHostToDevice));
    if (status != 0) return status;
    status = check_cuda(cudaMemset(device_score_accum, 0, total * sizeof(float)));
    if (status != 0) return status;

    pad_reflect_and_mask_kernel<<<grid2d, block2d>>>(
        device_image,
        device_circle_mask,
        width,
        height,
        padded_width,
        padded_height,
        device_approx_a
    );
    status = check_launch();
    if (status != 0) return status;

    float* approx_in = device_approx_a;
    float* approx_out = device_approx_b;
    for (int level_idx = 0; level_idx < active_level; ++level_idx) {
        const int stride = 1 << level_idx;
        convolve_rows_kernel<<<grid2d, block2d>>>(
            approx_in,
            device_low_rows,
            device_high_rows,
            padded_width,
            padded_height,
            stride
        );
        status = check_launch();
        if (status != 0) return status;

        convolve_cols_kernel<<<grid2d, block2d>>>(
            device_low_rows,
            device_high_rows,
            approx_out,
            device_lh,
            device_hl,
            device_hh,
            padded_width,
            padded_height,
            stride
        );
        status = check_launch();
        if (status != 0) return status;

        status = check_cuda(cudaMemset(device_sum, 0, sizeof(double)));
        if (status != 0) return status;
        status = check_cuda(cudaMemset(device_sq_sum, 0, sizeof(double)));
        if (status != 0) return status;
        status = check_cuda(cudaMemset(device_count, 0, sizeof(unsigned long long)));
        if (status != 0) return status;

        reduce_response_stats_kernel<<<grid1d, kBlockSize1D>>>(
            device_lh,
            device_hl,
            device_hh,
            device_image,
            device_circle_mask,
            width,
            height,
            padded_width,
            device_sum,
            device_sq_sum,
            device_count
        );
        status = check_launch();
        if (status != 0) return status;

        compute_threshold_kernel<<<1, 1>>>(
            device_sum,
            device_sq_sum,
            device_count,
            threshold_sigma,
            device_threshold
        );
        status = check_launch();
        if (status != 0) return status;

        update_score_accum_kernel<<<grid1d, kBlockSize1D>>>(
            device_lh,
            device_hl,
            device_hh,
            device_image,
            device_circle_mask,
            width,
            height,
            padded_width,
            device_threshold,
            device_score_accum
        );
        status = check_launch();
        if (status != 0) return status;

        float* temp = approx_in;
        approx_in = approx_out;
        approx_out = temp;
    }

    finalize_outputs_kernel<<<grid1d, kBlockSize1D>>>(
        device_score_accum,
        device_image,
        device_circle_mask,
        width * height,
        device_binary,
        device_score
    );
    status = check_launch();
    if (status != 0) return status;

    status = check_cuda(cudaMemcpy(binary_out, device_binary, total_bytes, cudaMemcpyDeviceToHost));
    if (status != 0) return status;
    status = check_cuda(cudaMemcpy(score_out, device_score, total_bytes, cudaMemcpyDeviceToHost));
    if (status != 0) return status;
    return 0;
}

}  // namespace

extern "C" int wavelet_cuda_process_u8(
    const uint8_t* image_ptr,
    const uint8_t* circle_mask_ptr,
    int width,
    int height,
    int level,
    float threshold_sigma,
    uint8_t* binary_out,
    uint8_t* score_out
) {
    if (image_ptr == nullptr || circle_mask_ptr == nullptr || binary_out == nullptr || score_out == nullptr) {
        return 1;
    }
    if (width <= 0 || height <= 0) {
        return 2;
    }

    const int active_level = level > 0 ? level : 1;
    const int scale = 1 << active_level;
    const int padded_width = ((width + scale - 1) / scale) * scale;
    const int padded_height = ((height + scale - 1) / scale) * scale;
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t total_bytes = total * sizeof(uint8_t);
    const size_t padded_total = static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height);
    const size_t padded_bytes = padded_total * sizeof(float);

    uint8_t* device_image = nullptr;
    uint8_t* device_circle_mask = nullptr;
    uint8_t* device_binary = nullptr;
    uint8_t* device_score = nullptr;
    float* device_approx_a = nullptr;
    float* device_approx_b = nullptr;
    float* device_low_rows = nullptr;
    float* device_high_rows = nullptr;
    float* device_lh = nullptr;
    float* device_hl = nullptr;
    float* device_hh = nullptr;
    float* device_score_accum = nullptr;
    double* device_sum = nullptr;
    double* device_sq_sum = nullptr;
    unsigned long long* device_count = nullptr;
    float* device_threshold = nullptr;

    int status = 0;

    status = check_cuda(cudaMalloc(&device_image, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_circle_mask, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_binary, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_score, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_approx_a, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_approx_b, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_low_rows, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_high_rows, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_lh, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_hl, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_hh, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_score_accum, total * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_sum, sizeof(double)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_sq_sum, sizeof(double)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_count, sizeof(unsigned long long)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_threshold, sizeof(float)));
    if (status != 0) goto cleanup;

    status = check_cuda(cudaMemcpy(device_circle_mask, circle_mask_ptr, total_bytes, cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;
    status = process_one_image_u8(
        image_ptr,
        device_circle_mask,
        width,
        height,
        level,
        threshold_sigma,
        device_image,
        device_binary,
        device_score,
        device_approx_a,
        device_approx_b,
        device_low_rows,
        device_high_rows,
        device_lh,
        device_hl,
        device_hh,
        device_score_accum,
        device_sum,
        device_sq_sum,
        device_count,
        device_threshold,
        binary_out,
        score_out
    );
    if (status != 0) goto cleanup;

cleanup:
    cudaFree(device_threshold);
    cudaFree(device_count);
    cudaFree(device_sq_sum);
    cudaFree(device_sum);
    cudaFree(device_score_accum);
    cudaFree(device_hh);
    cudaFree(device_hl);
    cudaFree(device_lh);
    cudaFree(device_high_rows);
    cudaFree(device_low_rows);
    cudaFree(device_approx_b);
    cudaFree(device_approx_a);
    cudaFree(device_score);
    cudaFree(device_binary);
    cudaFree(device_circle_mask);
    cudaFree(device_image);
    return status;
}

extern "C" int wavelet_cuda_process_batch_u8(
    const uint8_t* images_ptr,
    const uint8_t* circle_mask_ptr,
    int batch,
    int width,
    int height,
    int level,
    float threshold_sigma,
    uint8_t* binary_out,
    uint8_t* score_out
) {
    if (images_ptr == nullptr || circle_mask_ptr == nullptr || binary_out == nullptr || score_out == nullptr) {
        return 1;
    }
    if (batch <= 0 || width <= 0 || height <= 0) {
        return 2;
    }

    const int active_level = level > 0 ? level : 1;
    const int scale = 1 << active_level;
    const int padded_width = ((width + scale - 1) / scale) * scale;
    const int padded_height = ((height + scale - 1) / scale) * scale;
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t total_bytes = total * sizeof(uint8_t);
    const size_t padded_total = static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height);
    const size_t padded_bytes = padded_total * sizeof(float);

    uint8_t* device_image = nullptr;
    uint8_t* device_circle_mask = nullptr;
    uint8_t* device_binary = nullptr;
    uint8_t* device_score = nullptr;
    float* device_approx_a = nullptr;
    float* device_approx_b = nullptr;
    float* device_low_rows = nullptr;
    float* device_high_rows = nullptr;
    float* device_lh = nullptr;
    float* device_hl = nullptr;
    float* device_hh = nullptr;
    float* device_score_accum = nullptr;
    double* device_sum = nullptr;
    double* device_sq_sum = nullptr;
    unsigned long long* device_count = nullptr;
    float* device_threshold = nullptr;

    int status = 0;
    status = check_cuda(cudaMalloc(&device_image, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_circle_mask, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_binary, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_score, total_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_approx_a, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_approx_b, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_low_rows, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_high_rows, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_lh, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_hl, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_hh, padded_bytes));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_score_accum, total * sizeof(float)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_sum, sizeof(double)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_sq_sum, sizeof(double)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_count, sizeof(unsigned long long)));
    if (status != 0) goto cleanup;
    status = check_cuda(cudaMalloc(&device_threshold, sizeof(float)));
    if (status != 0) goto cleanup;

    status = check_cuda(cudaMemcpy(device_circle_mask, circle_mask_ptr, total_bytes, cudaMemcpyHostToDevice));
    if (status != 0) goto cleanup;

    for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const uint8_t* image_ptr = images_ptr + static_cast<size_t>(batch_idx) * total;
        uint8_t* binary_ptr = binary_out + static_cast<size_t>(batch_idx) * total;
        uint8_t* score_ptr = score_out + static_cast<size_t>(batch_idx) * total;
        status = process_one_image_u8(
            image_ptr,
            device_circle_mask,
            width,
            height,
            level,
            threshold_sigma,
            device_image,
            device_binary,
            device_score,
            device_approx_a,
            device_approx_b,
            device_low_rows,
            device_high_rows,
            device_lh,
            device_hl,
            device_hh,
            device_score_accum,
            device_sum,
            device_sq_sum,
            device_count,
            device_threshold,
            binary_ptr,
            score_ptr
        );
        if (status != 0) goto cleanup;
    }

cleanup:
    cudaFree(device_threshold);
    cudaFree(device_count);
    cudaFree(device_sq_sum);
    cudaFree(device_sum);
    cudaFree(device_score_accum);
    cudaFree(device_hh);
    cudaFree(device_hl);
    cudaFree(device_lh);
    cudaFree(device_high_rows);
    cudaFree(device_low_rows);
    cudaFree(device_approx_b);
    cudaFree(device_approx_a);
    cudaFree(device_score);
    cudaFree(device_binary);
    cudaFree(device_circle_mask);
    cudaFree(device_image);
    return status;
}
