#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

__global__ void histogram_nonzero_kernel(
    const uint8_t* image,
    const uint8_t* circle_mask,
    int total,
    unsigned int* histogram
) {
    __shared__ unsigned int local_hist[256];
    const int tid = threadIdx.x;
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    while (index < total) {
        const uint8_t pixel = image[index];
        if (circle_mask[index] && pixel > 0) {
            atomicAdd(&local_hist[pixel], 1U);
        }
        index += stride;
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&histogram[tid], local_hist[tid]);
    }
}

__global__ void threshold_mask_kernel(
    const uint8_t* image,
    const uint8_t* circle_mask,
    int total,
    uint8_t threshold,
    uint8_t* binary_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const uint8_t pixel = image[index];
    binary_out[index] = (circle_mask[index] && pixel > threshold) ? 1 : 0;
}

__global__ void binary_morph_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    const uint8_t* kernel,
    int kernel_size,
    int dilate
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int radius = kernel_size / 2;
    uint8_t value = dilate ? 0 : 1;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            if (!kernel[ky * kernel_size + kx]) {
                continue;
            }
            const int nx = x + kx - radius;
            const int ny = y + ky - radius;
            const bool inside = nx >= 0 && ny >= 0 && nx < width && ny < height;
            if (dilate) {
                if (inside && input[ny * width + nx]) {
                    value = 1;
                    ky = kernel_size;
                    break;
                }
            } else if (!inside || !input[ny * width + nx]) {
                value = 0;
                ky = kernel_size;
                break;
            }
        }
    }

    output[y * width + x] = value;
}

inline int check_cuda(cudaError_t status) {
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

uint8_t otsu_from_histogram(const unsigned int* hist) {
    double total_sum = 0.0;
    unsigned int count = 0;
    for (int i = 0; i < 256; ++i) {
        total_sum += static_cast<double>(i) * static_cast<double>(hist[i]);
        count += hist[i];
    }
    if (count < 100) {
        return 0;
    }

    const double total_weight = static_cast<double>(count);
    double background_weight = 0.0;
    double background_sum = 0.0;
    double best_variance = -1.0;
    int best_threshold = 0;

    for (int threshold = 0; threshold < 256; ++threshold) {
        background_weight += static_cast<double>(hist[threshold]);
        if (background_weight == 0.0) {
            continue;
        }
        const double foreground_weight = total_weight - background_weight;
        if (foreground_weight == 0.0) {
            break;
        }
        background_sum += static_cast<double>(threshold) * static_cast<double>(hist[threshold]);
        const double background_mean = background_sum / background_weight;
        const double foreground_mean = (total_sum - background_sum) / foreground_weight;
        const double mean_delta = background_mean - foreground_mean;
        const double variance = background_weight * foreground_weight * mean_delta * mean_delta;
        if (variance > best_variance) {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    return static_cast<uint8_t>(best_threshold);
}

std::vector<uint8_t> ellipse_kernel(int size) {
    if (size <= 1) {
        return std::vector<uint8_t>{1};
    }
    const int radius = size / 2;
    std::vector<uint8_t> kernel(size * size, 0);
    for (int dy = -radius; dy <= radius; ++dy) {
        const float radius_f = static_cast<float>(radius);
        const float dy_f = static_cast<float>(dy);
        const int dx = static_cast<int>(std::lround(
            radius_f * std::sqrt(1.0f - (dy_f * dy_f) / (radius_f * radius_f))
        ));
        for (int dx_i = -dx; dx_i <= dx; ++dx_i) {
            kernel[(dy + radius) * size + (dx_i + radius)] = 1;
        }
    }
    return kernel;
}

int apply_binary_morph(
    uint8_t* scratch_a,
    uint8_t* scratch_b,
    int width,
    int height,
    int kernel_size,
    int iterations,
    int dilate
) {
    if (iterations <= 0) {
        return 0;
    }

    const std::vector<uint8_t> host_kernel = ellipse_kernel(kernel_size);
    uint8_t* device_kernel = nullptr;
    if (check_cuda(cudaMalloc(&device_kernel, host_kernel.size())) != 0) {
        return 1;
    }
    const int copy_status = check_cuda(cudaMemcpy(
        device_kernel,
        host_kernel.data(),
        host_kernel.size(),
        cudaMemcpyHostToDevice
    ));
    if (copy_status != 0) {
        cudaFree(device_kernel);
        return copy_status;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    uint8_t* src = scratch_a;
    uint8_t* dst = scratch_b;
    for (int iter = 0; iter < iterations; ++iter) {
        binary_morph_kernel<<<grid, block>>>(src, dst, width, height, device_kernel, kernel_size, dilate);
        const int sync_status = check_cuda(cudaDeviceSynchronize());
        if (sync_status != 0) {
            cudaFree(device_kernel);
            return sync_status;
        }
        uint8_t* temp = src;
        src = dst;
        dst = temp;
    }

    if (src != scratch_a) {
        const int copy_back_status = check_cuda(cudaMemcpy(
            scratch_a,
            src,
            static_cast<size_t>(width) * static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice
        ));
        if (copy_back_status != 0) {
            cudaFree(device_kernel);
            return copy_back_status;
        }
    }

    cudaFree(device_kernel);
    return 0;
}

}  // namespace

extern "C" int morph_cuda_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        return 0;
    }
    return count;
}

extern "C" int morph_cuda_process_u8(
    const uint8_t* image_ptr,
    const uint8_t* circle_mask_ptr,
    int width,
    int height,
    int opening_kernel_size,
    int opening_iterations,
    int closing_kernel_size,
    int closing_iterations,
    uint8_t* threshold_out,
    uint8_t* binary_out
) {
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    uint8_t* device_image = nullptr;
    uint8_t* device_circle_mask = nullptr;
    uint8_t* device_a = nullptr;
    uint8_t* device_b = nullptr;
    unsigned int* device_histogram = nullptr;

    int status = check_cuda(cudaMalloc(&device_image, total));
    if (status != 0) return status;
    status = check_cuda(cudaMalloc(&device_circle_mask, total));
    if (status != 0) return status;
    status = check_cuda(cudaMalloc(&device_a, total));
    if (status != 0) return status;
    status = check_cuda(cudaMalloc(&device_b, total));
    if (status != 0) return status;
    status = check_cuda(cudaMalloc(&device_histogram, 256 * sizeof(unsigned int)));
    if (status != 0) return status;

    status = check_cuda(cudaMemcpy(device_image, image_ptr, total, cudaMemcpyHostToDevice));
    if (status != 0) return status;
    status = check_cuda(cudaMemcpy(device_circle_mask, circle_mask_ptr, total, cudaMemcpyHostToDevice));
    if (status != 0) return status;
    status = check_cuda(cudaMemset(device_histogram, 0, 256 * sizeof(unsigned int)));
    if (status != 0) return status;

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    histogram_nonzero_kernel<<<blocks, threads>>>(device_image, device_circle_mask, static_cast<int>(total), device_histogram);
    status = check_cuda(cudaDeviceSynchronize());
    if (status != 0) return status;

    unsigned int host_histogram[256];
    status = check_cuda(cudaMemcpy(host_histogram, device_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (status != 0) return status;

    const uint8_t threshold = otsu_from_histogram(host_histogram);
    *threshold_out = threshold;
    threshold_mask_kernel<<<blocks, threads>>>(device_image, device_circle_mask, static_cast<int>(total), threshold, device_a);
    status = check_cuda(cudaDeviceSynchronize());
    if (status != 0) return status;

    status = apply_binary_morph(device_a, device_b, width, height, opening_kernel_size, opening_iterations, 0);
    if (status != 0) return status;
    status = apply_binary_morph(device_a, device_b, width, height, opening_kernel_size, opening_iterations, 1);
    if (status != 0) return status;
    status = apply_binary_morph(device_a, device_b, width, height, closing_kernel_size, closing_iterations, 1);
    if (status != 0) return status;
    status = apply_binary_morph(device_a, device_b, width, height, closing_kernel_size, closing_iterations, 0);
    if (status != 0) return status;

    status = check_cuda(cudaMemcpy(binary_out, device_a, total, cudaMemcpyDeviceToHost));
    if (status != 0) return status;

    cudaFree(device_histogram);
    cudaFree(device_b);
    cudaFree(device_a);
    cudaFree(device_circle_mask);
    cudaFree(device_image);
    return 0;
}
