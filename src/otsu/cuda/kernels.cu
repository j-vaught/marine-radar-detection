#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <new>
#include <vector>

struct RawDetectionCandidate {
    unsigned int bbox_xyxy[4];
    unsigned int area;
    float centroid_x;
    float centroid_y;
    float raw_score;
};

namespace {

constexpr unsigned int kReqUnset = 0xFFFFFFFFu;

struct OtsuCudaCache {
    int roi_width = 0;
    int roi_height = 0;
    int roi_pixels = 0;
    uint8_t* image = nullptr;
    uint8_t* binary = nullptr;
    uint16_t* threshold_map = nullptr;
};

struct OtsuCudaWorkspace {
    int capacity_pixels = 0;
    int label_slots = 0;
    int detections_capacity = 0;
    uint8_t* binary_work = nullptr;
    unsigned int* labels_a = nullptr;
    unsigned int* labels_b = nullptr;
    int* changed = nullptr;
    unsigned int* area = nullptr;
    unsigned int* min_x = nullptr;
    unsigned int* min_y = nullptr;
    unsigned int* max_x = nullptr;
    unsigned int* max_y = nullptr;
    unsigned int* sum_x = nullptr;
    unsigned int* sum_y = nullptr;
    unsigned int* peak = nullptr;
    unsigned int* active_labels = nullptr;
    unsigned int* active_count = nullptr;
    RawDetectionCandidate* detections = nullptr;
    unsigned int* detections_count = nullptr;

    void release() {
        cudaFree(binary_work);
        cudaFree(labels_a);
        cudaFree(labels_b);
        cudaFree(changed);
        cudaFree(area);
        cudaFree(min_x);
        cudaFree(min_y);
        cudaFree(max_x);
        cudaFree(max_y);
        cudaFree(sum_x);
        cudaFree(sum_y);
        cudaFree(peak);
        cudaFree(active_labels);
        cudaFree(active_count);
        cudaFree(detections);
        cudaFree(detections_count);
        binary_work = nullptr;
        labels_a = nullptr;
        labels_b = nullptr;
        changed = nullptr;
        area = nullptr;
        min_x = nullptr;
        min_y = nullptr;
        max_x = nullptr;
        max_y = nullptr;
        sum_x = nullptr;
        sum_y = nullptr;
        peak = nullptr;
        active_labels = nullptr;
        active_count = nullptr;
        detections = nullptr;
        detections_count = nullptr;
        capacity_pixels = 0;
        label_slots = 0;
        detections_capacity = 0;
    }
};

OtsuCudaWorkspace g_workspace;
std::mutex g_workspace_mutex;

inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline bool check(cudaError_t status) {
    return status == cudaSuccess;
}

bool otsu_threshold_from_hist_host(const unsigned int* histogram, unsigned int count, uint16_t* out) {
    if (count == 0) {
        return false;
    }
    double total_sum = 0.0;
    for (int value = 0; value < 256; ++value) {
        total_sum += static_cast<double>(value) * static_cast<double>(histogram[value]);
    }

    double background_weight = 0.0;
    double background_sum = 0.0;
    double best_variance = -1.0;
    int best_threshold = 0;
    const double total_weight = static_cast<double>(count);

    for (int threshold = 0; threshold < 256; ++threshold) {
        background_weight += static_cast<double>(histogram[threshold]);
        if (background_weight == 0.0) {
            continue;
        }
        const double foreground_weight = total_weight - background_weight;
        if (foreground_weight == 0.0) {
            break;
        }
        background_sum += static_cast<double>(threshold) * static_cast<double>(histogram[threshold]);
        const double background_mean = background_sum / background_weight;
        const double foreground_mean = (total_sum - background_sum) / foreground_weight;
        const double mean_delta = background_mean - foreground_mean;
        const double variance = background_weight * foreground_weight * mean_delta * mean_delta;
        if (variance > best_variance) {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    *out = static_cast<uint16_t>(best_threshold);
    return true;
}

__device__ bool otsu_threshold_from_hist_device(const unsigned int* histogram, unsigned int count, uint16_t* out) {
    if (count == 0) {
        return false;
    }
    double total_sum = 0.0;
    for (int value = 0; value < 256; ++value) {
        total_sum += static_cast<double>(value) * static_cast<double>(histogram[value]);
    }

    double background_weight = 0.0;
    double background_sum = 0.0;
    double best_variance = -1.0;
    int best_threshold = 0;
    const double total_weight = static_cast<double>(count);

    for (int threshold = 0; threshold < 256; ++threshold) {
        background_weight += static_cast<double>(histogram[threshold]);
        if (background_weight == 0.0) {
            continue;
        }
        const double foreground_weight = total_weight - background_weight;
        if (foreground_weight == 0.0) {
            break;
        }
        background_sum += static_cast<double>(threshold) * static_cast<double>(histogram[threshold]);
        const double background_mean = background_sum / background_weight;
        const double foreground_mean = (total_sum - background_sum) / foreground_weight;
        const double mean_delta = background_mean - foreground_mean;
        const double variance = background_weight * foreground_weight * mean_delta * mean_delta;
        if (variance > best_variance) {
            best_variance = variance;
            best_threshold = threshold;
        }
    }

    *out = static_cast<uint16_t>(best_threshold);
    return true;
}

__global__ void masked_histogram_roi_kernel(
    const uint8_t* image,
    const uint8_t* mask,
    int image_width,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    unsigned int* histogram) {
    __shared__ unsigned int local_hist[256];
    for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
        local_hist[idx] = 0;
    }
    __syncthreads();

    const int total = roi_width * roi_height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    while (idx < total) {
        const int x = idx % roi_width;
        const int y = idx / roi_width;
        const int src = (roi_y0 + y) * image_width + roi_x0 + x;
        const uint8_t pixel = image[src];
        if (mask[src] != 0 && pixel > 0) {
            atomicAdd(&local_hist[pixel], 1u);
        }
        idx += stride;
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x) {
        atomicAdd(&histogram[bin], local_hist[bin]);
    }
}

__global__ void apply_global_threshold_roi_kernel(
    const uint8_t* image,
    const uint8_t* mask,
    int image_width,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    uint16_t threshold,
    uint8_t* binary_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = roi_width * roi_height;
    if (idx >= total) {
        return;
    }
    const int x = idx % roi_width;
    const int y = idx / roi_width;
    const int src = (roi_y0 + y) * image_width + roi_x0 + x;
    const uint8_t pixel = image[src];
    binary_out[idx] = static_cast<uint8_t>(mask[src] != 0 && pixel > 0 && pixel > threshold);
}

__global__ void ssotsu_seed_blocks_kernel(
    const uint8_t* image,
    const uint8_t* mask,
    int image_width,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    int block_size,
    float seed_threshold_factor,
    float region_grow_threshold,
    uint16_t* threshold_map_out,
    unsigned int* requirements_out) {
    const int block_x0 = blockIdx.x * block_size;
    const int block_y0 = blockIdx.y * block_size;
    if (block_x0 >= roi_width || block_y0 >= roi_height) {
        return;
    }

    const int tile_width = min(block_size, roi_width - block_x0);
    const int tile_height = min(block_size, roi_height - block_y0);
    const int tile_pixels = tile_width * tile_height;

    __shared__ unsigned int histogram[256];
    __shared__ unsigned int nz_count;
    __shared__ unsigned long long sum;
    __shared__ unsigned long long sq_sum;
    __shared__ uint16_t seed_threshold;
    __shared__ int valid_tile;

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        histogram[i] = 0;
    }
    if (threadIdx.x == 0) {
        nz_count = 0;
        sum = 0;
        sq_sum = 0;
        seed_threshold = 1;
        valid_tile = 0;
    }
    __syncthreads();

    for (int local = threadIdx.x; local < tile_pixels; local += blockDim.x) {
        const int lx = local % tile_width;
        const int ly = local / tile_width;
        const int gx = roi_x0 + block_x0 + lx;
        const int gy = roi_y0 + block_y0 + ly;
        const int src = gy * image_width + gx;
        const uint8_t pixel = image[src];
        if (mask[src] != 0 && pixel > 0) {
            atomicAdd(&histogram[pixel], 1u);
            atomicAdd(&nz_count, 1u);
            atomicAdd(&sum, static_cast<unsigned long long>(pixel));
            atomicAdd(&sq_sum, static_cast<unsigned long long>(pixel) * static_cast<unsigned long long>(pixel));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (nz_count >= 10) {
            const double mean = static_cast<double>(sum) / static_cast<double>(nz_count);
            const double variance = fmax(
                static_cast<double>(sq_sum) / static_cast<double>(nz_count) - mean * mean,
                0.0);
            if (sqrt(variance) >= 1.0) {
                uint16_t local_threshold = 0;
                if (otsu_threshold_from_hist_device(histogram, nz_count, &local_threshold)) {
                    seed_threshold = static_cast<uint16_t>(fmax(1.0f, ceilf(local_threshold * seed_threshold_factor)));
                    valid_tile = 1;
                }
            }
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < tile_pixels; local += blockDim.x) {
        const int lx = local % tile_width;
        const int ly = local / tile_width;
        const int roi_idx = (block_y0 + ly) * roi_width + (block_x0 + lx);
        const int gx = roi_x0 + block_x0 + lx;
        const int gy = roi_y0 + block_y0 + ly;
        const int src = gy * image_width + gx;
        const uint8_t pixel = image[src];
        threshold_map_out[roi_idx] = seed_threshold;
        if (valid_tile && mask[src] != 0 && pixel > 0 && static_cast<uint16_t>(pixel) > seed_threshold) {
            const unsigned int requirement = static_cast<unsigned int>(
                fminf(255.0f, fmaxf(1.0f, ceilf(pixel * region_grow_threshold))));
            requirements_out[roi_idx] = requirement;
        }
    }
}

__global__ void ssotsu_propagate_kernel(
    const uint8_t* image,
    const uint8_t* mask,
    int image_width,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    unsigned int* requirements,
    int* changed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = roi_width * roi_height;
    if (idx >= total) {
        return;
    }

    const unsigned int requirement = requirements[idx];
    if (requirement == kReqUnset) {
        return;
    }

    const int x = idx % roi_width;
    const int y = idx / roi_width;

    const int nx[4] = {x - 1, x + 1, x, x};
    const int ny[4] = {y, y, y - 1, y + 1};
    const bool valid[4] = {x > 0, x + 1 < roi_width, y > 0, y + 1 < roi_height};

    for (int dir = 0; dir < 4; ++dir) {
        if (!valid[dir]) {
            continue;
        }
        const int rx = nx[dir];
        const int ry = ny[dir];
        const int nidx = ry * roi_width + rx;
        const int src = (roi_y0 + ry) * image_width + roi_x0 + rx;
        const uint8_t pixel = image[src];
        if (mask[src] == 0 || pixel == 0 || pixel < requirement) {
            continue;
        }
        const unsigned int old = atomicMin(&requirements[nidx], requirement);
        if (old > requirement) {
            *changed = 1;
        }
    }
}

__global__ void requirements_to_binary_kernel(
    const unsigned int* requirements,
    int count,
    uint8_t* binary_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    binary_out[idx] = static_cast<uint8_t>(requirements[idx] != kReqUnset);
}

__device__ int ellipse_dx_for_dy(int radius, int dy) {
    if (radius <= 0) {
        return 0;
    }
    const float radius_f = static_cast<float>(radius);
    const float dy_f = static_cast<float>(dy);
    const float term = fmaxf(0.0f, 1.0f - (dy_f * dy_f) / (radius_f * radius_f));
    return static_cast<int>(roundf(radius_f * sqrtf(term)));
}

__global__ void dilate_binary_kernel(
    const uint8_t* binary_in,
    int width,
    int height,
    int kernel_size,
    uint8_t* binary_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }
    const int x = idx % width;
    const int y = idx / width;
    const int radius = kernel_size / 2;
    uint8_t value = 0;
    for (int dy = -radius; dy <= radius && value == 0; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) {
            continue;
        }
        const int dx_limit = ellipse_dx_for_dy(radius, dy);
        for (int dx = -dx_limit; dx <= dx_limit; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= width) {
                continue;
            }
            if (binary_in[ny * width + nx] != 0) {
                value = 1;
                break;
            }
        }
    }
    binary_out[idx] = value;
}

__global__ void init_labels_kernel(
    const uint8_t* binary,
    int count,
    unsigned int* labels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    labels[idx] = binary[idx] != 0 ? static_cast<unsigned int>(idx + 1) : 0u;
}

__global__ void propagate_labels_kernel(
    const uint8_t* binary,
    int width,
    int height,
    const unsigned int* labels_in,
    unsigned int* labels_out,
    int* changed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }
    if (binary[idx] == 0) {
        labels_out[idx] = 0u;
        return;
    }

    const int x = idx % width;
    const int y = idx / width;
    unsigned int best = labels_in[idx];
    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= width) {
                continue;
            }
            const int nidx = ny * width + nx;
            if (binary[nidx] == 0) {
                continue;
            }
            const unsigned int label = labels_in[nidx];
            if (label != 0u && label < best) {
                best = label;
            }
        }
    }
    labels_out[idx] = best;
    if (best != labels_in[idx]) {
        *changed = 1;
    }
}

__global__ void reduce_components_kernel(
    const uint8_t* binary,
    const unsigned int* labels,
    const uint8_t* image,
    int width,
    int height,
    unsigned int* area,
    unsigned int* min_x,
    unsigned int* min_y,
    unsigned int* max_x,
    unsigned int* max_y,
    unsigned int* sum_x,
    unsigned int* sum_y,
    unsigned int* peak,
    unsigned int* active_labels,
    unsigned int* active_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total || binary[idx] == 0) {
        return;
    }

    const unsigned int label = labels[idx];
    if (label == 0u) {
        return;
    }

    const unsigned int x = static_cast<unsigned int>(idx % width);
    const unsigned int y = static_cast<unsigned int>(idx / width);
    const unsigned int pixel = static_cast<unsigned int>(image[idx]);

    const unsigned int old_area = atomicAdd(&area[label], 1u);
    if (old_area == 0u) {
        const unsigned int slot = atomicAdd(active_count, 1u);
        active_labels[slot] = label;
    }
    atomicMin(&min_x[label], x);
    atomicMin(&min_y[label], y);
    atomicMax(&max_x[label], x + 1u);
    atomicMax(&max_y[label], y + 1u);
    atomicAdd(&sum_x[label], x);
    atomicAdd(&sum_y[label], y);
    atomicMax(&peak[label], pixel);
}

__global__ void gather_components_kernel(
    const unsigned int* active_labels,
    unsigned int active_count,
    const unsigned int* area,
    const unsigned int* min_x,
    const unsigned int* min_y,
    const unsigned int* max_x,
    const unsigned int* max_y,
    const unsigned int* sum_x,
    const unsigned int* sum_y,
    const unsigned int* peak,
    const uint16_t* threshold_map,
    uint16_t threshold_constant,
    int width,
    int height,
    unsigned int min_area,
    unsigned int max_area,
    RawDetectionCandidate* detections_out,
    unsigned int* detections_count_out) {
    const unsigned int idx = static_cast<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= active_count) {
        return;
    }

    const unsigned int label = active_labels[idx];
    const unsigned int component_area = area[label];
    if (component_area < min_area || component_area > max_area) {
        return;
    }

    const float centroid_x = static_cast<float>(sum_x[label]) / static_cast<float>(component_area);
    const float centroid_y = static_cast<float>(sum_y[label]) / static_cast<float>(component_area);
    const int cx = max(0, min(width - 1, static_cast<int>(roundf(centroid_x))));
    const int cy = max(0, min(height - 1, static_cast<int>(roundf(centroid_y))));
    const float threshold = threshold_map != nullptr
        ? fmaxf(1.0f, static_cast<float>(threshold_map[cy * width + cx]))
        : fmaxf(1.0f, static_cast<float>(threshold_constant));

    const unsigned int slot = atomicAdd(detections_count_out, 1u);
    detections_out[slot].bbox_xyxy[0] = min_x[label];
    detections_out[slot].bbox_xyxy[1] = min_y[label];
    detections_out[slot].bbox_xyxy[2] = max_x[label];
    detections_out[slot].bbox_xyxy[3] = max_y[label];
    detections_out[slot].area = component_area;
    detections_out[slot].centroid_x = centroid_x;
    detections_out[slot].centroid_y = centroid_y;
    detections_out[slot].raw_score = static_cast<float>(peak[label]) / threshold;
}

bool ensure_workspace(int roi_pixels) {
    if (roi_pixels <= 0) {
        return true;
    }
    if (g_workspace.capacity_pixels >= roi_pixels &&
        g_workspace.detections_capacity >= roi_pixels) {
        return true;
    }

    g_workspace.release();

    g_workspace.capacity_pixels = roi_pixels;
    g_workspace.label_slots = roi_pixels + 1;
    g_workspace.detections_capacity = roi_pixels;

    const bool ok =
        check(cudaMalloc(&g_workspace.binary_work, roi_pixels * sizeof(uint8_t))) &&
        check(cudaMalloc(&g_workspace.labels_a, roi_pixels * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.labels_b, roi_pixels * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.changed, sizeof(int))) &&
        check(cudaMalloc(&g_workspace.area, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.min_x, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.min_y, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.max_x, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.max_y, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.sum_x, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.sum_y, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.peak, g_workspace.label_slots * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.active_labels, roi_pixels * sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.active_count, sizeof(unsigned int))) &&
        check(cudaMalloc(&g_workspace.detections, roi_pixels * sizeof(RawDetectionCandidate))) &&
        check(cudaMalloc(&g_workspace.detections_count, sizeof(unsigned int)));

    if (!ok) {
        g_workspace.release();
        return false;
    }
    return true;
}

void destroy_cache(OtsuCudaCache* cache) {
    if (!cache) {
        return;
    }
    cudaFree(cache->image);
    cudaFree(cache->binary);
    cudaFree(cache->threshold_map);
    delete cache;
}

int postprocess_cached_impl(
    const OtsuCudaCache* cache,
    uint16_t threshold_constant,
    int dilation,
    unsigned int min_area,
    unsigned int max_area,
    RawDetectionCandidate* detections_out,
    int max_detections,
    int* detections_count_out) {
    if (!cache || !detections_out || !detections_count_out) {
        return 1;
    }
    if (cache->roi_pixels <= 0 || max_detections <= 0) {
        *detections_count_out = 0;
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (!ensure_workspace(cache->roi_pixels)) {
        return 2;
    }

    const int threads = 256;
    const int blocks = ceil_div(cache->roi_pixels, threads);
    const uint8_t* active_binary = cache->binary;

    if (dilation > 0) {
        const int kernel_size = 2 * dilation + 1;
        dilate_binary_kernel<<<blocks, threads>>>(
            cache->binary,
            cache->roi_width,
            cache->roi_height,
            kernel_size,
            g_workspace.binary_work);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            return 3;
        }
        active_binary = g_workspace.binary_work;
    }

    init_labels_kernel<<<blocks, threads>>>(active_binary, cache->roi_pixels, g_workspace.labels_a);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        return 4;
    }

    const int max_iters = cache->roi_width + cache->roi_height + 8;
    for (int iter = 0; iter < max_iters; ++iter) {
        if (!check(cudaMemset(g_workspace.changed, 0, sizeof(int)))) {
            return 5;
        }
        propagate_labels_kernel<<<blocks, threads>>>(
            active_binary,
            cache->roi_width,
            cache->roi_height,
            g_workspace.labels_a,
            g_workspace.labels_b,
            g_workspace.changed);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            return 6;
        }
        int changed = 0;
        if (!check(cudaMemcpy(&changed, g_workspace.changed, sizeof(int), cudaMemcpyDeviceToHost))) {
            return 7;
        }
        std::swap(g_workspace.labels_a, g_workspace.labels_b);
        if (changed == 0) {
            break;
        }
    }

    if (!check(cudaMemset(g_workspace.area, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.min_x, 0xFF, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.min_y, 0xFF, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.max_x, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.max_y, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.sum_x, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.sum_y, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.peak, 0, g_workspace.label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.active_count, 0, sizeof(unsigned int))) ||
        !check(cudaMemset(g_workspace.detections_count, 0, sizeof(unsigned int)))) {
        return 8;
    }

    reduce_components_kernel<<<blocks, threads>>>(
        active_binary,
        g_workspace.labels_a,
        cache->image,
        cache->roi_width,
        cache->roi_height,
        g_workspace.area,
        g_workspace.min_x,
        g_workspace.min_y,
        g_workspace.max_x,
        g_workspace.max_y,
        g_workspace.sum_x,
        g_workspace.sum_y,
        g_workspace.peak,
        g_workspace.active_labels,
        g_workspace.active_count);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        return 9;
    }

    unsigned int active_count = 0u;
    if (!check(cudaMemcpy(&active_count, g_workspace.active_count, sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
        return 10;
    }
    if (active_count == 0u) {
        *detections_count_out = 0;
        return 0;
    }

    const int gather_blocks = ceil_div(static_cast<int>(active_count), threads);
    gather_components_kernel<<<gather_blocks, threads>>>(
        g_workspace.active_labels,
        active_count,
        g_workspace.area,
        g_workspace.min_x,
        g_workspace.min_y,
        g_workspace.max_x,
        g_workspace.max_y,
        g_workspace.sum_x,
        g_workspace.sum_y,
        g_workspace.peak,
        cache->threshold_map,
        threshold_constant,
        cache->roi_width,
        cache->roi_height,
        min_area,
        max_area,
        g_workspace.detections,
        g_workspace.detections_count);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        return 11;
    }

    unsigned int detections_count = 0u;
    if (!check(cudaMemcpy(
            &detections_count,
            g_workspace.detections_count,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost))) {
        return 12;
    }

    detections_count = std::min(detections_count, static_cast<unsigned int>(max_detections));
    if (detections_count > 0u &&
        !check(cudaMemcpy(
            detections_out,
            g_workspace.detections,
            detections_count * sizeof(RawDetectionCandidate),
            cudaMemcpyDeviceToHost))) {
        return 13;
    }

    *detections_count_out = static_cast<int>(detections_count);
    return 0;
}

}  // namespace

extern "C" int otsu_cuda_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int otsu_cuda_preprocess_global(
    const uint8_t* image_ptr,
    const uint8_t* circle_mask_ptr,
    int image_width,
    int image_height,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    uint16_t* threshold_out,
    uint8_t* binary_out) {
    if (!image_ptr || !circle_mask_ptr || !threshold_out || !binary_out) {
        return 1;
    }

    const int image_pixels = image_width * image_height;
    const int roi_pixels = roi_width * roi_height;
    uint8_t* d_image = nullptr;
    uint8_t* d_mask = nullptr;
    unsigned int* d_hist = nullptr;
    uint8_t* d_binary = nullptr;

    if (!check(cudaMalloc(&d_image, image_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_mask, image_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_hist, 256 * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_binary, roi_pixels * sizeof(uint8_t)))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 2;
    }

    if (!check(cudaMemcpy(d_image, image_ptr, image_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(d_mask, circle_mask_ptr, image_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice)) ||
        !check(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 3;
    }

    const int threads = 256;
    const int blocks = ceil_div(roi_pixels, threads);
    masked_histogram_roi_kernel<<<blocks, threads>>>(
        d_image, d_mask, image_width, roi_x0, roi_y0, roi_width, roi_height, d_hist);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 4;
    }

    unsigned int histogram[256];
    if (!check(cudaMemcpy(histogram, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 5;
    }
    unsigned int count = 0;
    for (unsigned int bin : histogram) {
        count += bin;
    }

    uint16_t threshold = 1;
    if (count >= 100 && otsu_threshold_from_hist_host(histogram, count, &threshold)) {
        apply_global_threshold_roi_kernel<<<blocks, threads>>>(
            d_image, d_mask, image_width, roi_x0, roi_y0, roi_width, roi_height, threshold, d_binary);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            cudaFree(d_image);
            cudaFree(d_mask);
            cudaFree(d_hist);
            cudaFree(d_binary);
            return 6;
        }
    } else if (!check(cudaMemset(d_binary, 0, roi_pixels * sizeof(uint8_t)))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 7;
    }

    if (!check(cudaMemcpy(binary_out, d_binary, roi_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_hist);
        cudaFree(d_binary);
        return 8;
    }

    *threshold_out = threshold;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_hist);
    cudaFree(d_binary);
    return 0;
}

extern "C" int otsu_cuda_preprocess_ssotsu(
    const uint8_t* image_ptr,
    const uint8_t* circle_mask_ptr,
    int image_width,
    int image_height,
    int roi_x0,
    int roi_y0,
    int roi_width,
    int roi_height,
    int block_size,
    float seed_threshold_factor,
    float region_grow_threshold,
    uint16_t* threshold_map_out,
    uint8_t* binary_out) {
    if (!image_ptr || !circle_mask_ptr || !threshold_map_out || !binary_out) {
        return 1;
    }

    const int image_pixels = image_width * image_height;
    const int roi_pixels = roi_width * roi_height;
    uint8_t* d_image = nullptr;
    uint8_t* d_mask = nullptr;
    uint16_t* d_threshold_map = nullptr;
    unsigned int* d_requirements = nullptr;
    uint8_t* d_binary = nullptr;
    int* d_changed = nullptr;

    if (!check(cudaMalloc(&d_image, image_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_mask, image_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_threshold_map, roi_pixels * sizeof(uint16_t))) ||
        !check(cudaMalloc(&d_requirements, roi_pixels * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_binary, roi_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_changed, sizeof(int)))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_threshold_map);
        cudaFree(d_requirements);
        cudaFree(d_binary);
        cudaFree(d_changed);
        return 2;
    }

    if (!check(cudaMemcpy(d_image, image_ptr, image_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(d_mask, circle_mask_ptr, image_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice)) ||
        !check(cudaMemset(d_requirements, 0xFF, roi_pixels * sizeof(unsigned int))) ||
        !check(cudaMemset(d_binary, 0, roi_pixels * sizeof(uint8_t)))) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_threshold_map);
        cudaFree(d_requirements);
        cudaFree(d_binary);
        cudaFree(d_changed);
        return 3;
    }

    const dim3 grid(ceil_div(roi_width, block_size), ceil_div(roi_height, block_size));
    ssotsu_seed_blocks_kernel<<<grid, 256>>>(
        d_image,
        d_mask,
        image_width,
        roi_x0,
        roi_y0,
        roi_width,
        roi_height,
        block_size,
        seed_threshold_factor,
        region_grow_threshold,
        d_threshold_map,
        d_requirements);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_threshold_map);
        cudaFree(d_requirements);
        cudaFree(d_binary);
        cudaFree(d_changed);
        return 4;
    }

    const int threads = 256;
    const int blocks = ceil_div(roi_pixels, threads);
    const int max_iters = roi_width + roi_height + 8;
    for (int iter = 0; iter < max_iters; ++iter) {
        if (!check(cudaMemset(d_changed, 0, sizeof(int)))) {
            cudaFree(d_image);
            cudaFree(d_mask);
            cudaFree(d_threshold_map);
            cudaFree(d_requirements);
            cudaFree(d_binary);
            cudaFree(d_changed);
            return 5;
        }
        ssotsu_propagate_kernel<<<blocks, threads>>>(
            d_image,
            d_mask,
            image_width,
            roi_x0,
            roi_y0,
            roi_width,
            roi_height,
            d_requirements,
            d_changed);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            cudaFree(d_image);
            cudaFree(d_mask);
            cudaFree(d_threshold_map);
            cudaFree(d_requirements);
            cudaFree(d_binary);
            cudaFree(d_changed);
            return 6;
        }
        int changed = 0;
        if (!check(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost))) {
            cudaFree(d_image);
            cudaFree(d_mask);
            cudaFree(d_threshold_map);
            cudaFree(d_requirements);
            cudaFree(d_binary);
            cudaFree(d_changed);
            return 7;
        }
        if (changed == 0) {
            break;
        }
    }

    requirements_to_binary_kernel<<<blocks, threads>>>(d_requirements, roi_pixels, d_binary);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cudaFree(d_image);
        cudaFree(d_mask);
        cudaFree(d_threshold_map);
        cudaFree(d_requirements);
        cudaFree(d_binary);
        cudaFree(d_changed);
        return 8;
    }

    const bool ok =
        check(cudaMemcpy(threshold_map_out, d_threshold_map, roi_pixels * sizeof(uint16_t), cudaMemcpyDeviceToHost)) &&
        check(cudaMemcpy(binary_out, d_binary, roi_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_threshold_map);
    cudaFree(d_requirements);
    cudaFree(d_binary);
    cudaFree(d_changed);
    return ok ? 0 : 9;
}

extern "C" int otsu_cuda_postprocess_u8(
    const uint8_t* roi_image_ptr,
    const uint8_t* binary_ptr,
    const uint16_t* threshold_map_ptr,
    uint16_t threshold_constant,
    int roi_width,
    int roi_height,
    int dilation,
    unsigned int min_area,
    unsigned int max_area,
    RawDetectionCandidate* detections_out,
    int max_detections,
    int* detections_count_out) {
    if (!roi_image_ptr || !binary_ptr || !detections_out || !detections_count_out) {
        return 1;
    }

    const int roi_pixels = roi_width * roi_height;
    const int label_slots = roi_pixels + 1;
    const int threads = 256;
    const int blocks = ceil_div(roi_pixels, threads);

    uint8_t* d_image = nullptr;
    uint8_t* d_binary_in = nullptr;
    uint8_t* d_binary_work = nullptr;
    uint8_t* d_binary_active = nullptr;
    uint16_t* d_threshold_map = nullptr;
    unsigned int* d_labels_a = nullptr;
    unsigned int* d_labels_b = nullptr;
    int* d_changed = nullptr;
    unsigned int* d_area = nullptr;
    unsigned int* d_min_x = nullptr;
    unsigned int* d_min_y = nullptr;
    unsigned int* d_max_x = nullptr;
    unsigned int* d_max_y = nullptr;
    unsigned int* d_sum_x = nullptr;
    unsigned int* d_sum_y = nullptr;
    unsigned int* d_peak = nullptr;
    unsigned int* d_active_labels = nullptr;
    unsigned int* d_active_count = nullptr;
    RawDetectionCandidate* d_detections = nullptr;
    unsigned int* d_detections_count = nullptr;

    const auto cleanup = [&]() {
        cudaFree(d_image);
        cudaFree(d_binary_in);
        cudaFree(d_binary_work);
        cudaFree(d_threshold_map);
        cudaFree(d_labels_a);
        cudaFree(d_labels_b);
        cudaFree(d_changed);
        cudaFree(d_area);
        cudaFree(d_min_x);
        cudaFree(d_min_y);
        cudaFree(d_max_x);
        cudaFree(d_max_y);
        cudaFree(d_sum_x);
        cudaFree(d_sum_y);
        cudaFree(d_peak);
        cudaFree(d_active_labels);
        cudaFree(d_active_count);
        cudaFree(d_detections);
        cudaFree(d_detections_count);
    };

    if (!check(cudaMalloc(&d_image, roi_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_binary_in, roi_pixels * sizeof(uint8_t))) ||
        !check(cudaMalloc(&d_labels_a, roi_pixels * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_labels_b, roi_pixels * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_changed, sizeof(int))) ||
        !check(cudaMalloc(&d_area, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_min_x, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_min_y, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_max_x, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_max_y, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_sum_x, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_sum_y, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_peak, label_slots * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_active_labels, roi_pixels * sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_active_count, sizeof(unsigned int))) ||
        !check(cudaMalloc(&d_detections_count, sizeof(unsigned int)))) {
        cleanup();
        return 2;
    }

    if (!check(cudaMemcpy(d_image, roi_image_ptr, roi_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(d_binary_in, binary_ptr, roi_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice))) {
        cleanup();
        return 3;
    }

    if (threshold_map_ptr != nullptr) {
        if (!check(cudaMalloc(&d_threshold_map, roi_pixels * sizeof(uint16_t))) ||
            !check(cudaMemcpy(d_threshold_map, threshold_map_ptr, roi_pixels * sizeof(uint16_t), cudaMemcpyHostToDevice))) {
            cleanup();
            return 4;
        }
    }

    d_binary_active = d_binary_in;
    if (dilation > 0) {
        const int kernel_size = 2 * dilation + 1;
        if (!check(cudaMalloc(&d_binary_work, roi_pixels * sizeof(uint8_t)))) {
            cleanup();
            return 5;
        }
        dilate_binary_kernel<<<blocks, threads>>>(d_binary_in, roi_width, roi_height, kernel_size, d_binary_work);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            cleanup();
            return 6;
        }
        d_binary_active = d_binary_work;
    }

    init_labels_kernel<<<blocks, threads>>>(d_binary_active, roi_pixels, d_labels_a);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cleanup();
        return 7;
    }

    const int max_iters = roi_width + roi_height + 8;
    for (int iter = 0; iter < max_iters; ++iter) {
        if (!check(cudaMemset(d_changed, 0, sizeof(int)))) {
            cleanup();
            return 8;
        }
        propagate_labels_kernel<<<blocks, threads>>>(
            d_binary_active, roi_width, roi_height, d_labels_a, d_labels_b, d_changed);
        if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
            cleanup();
            return 9;
        }
        int changed = 0;
        if (!check(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost))) {
            cleanup();
            return 10;
        }
        std::swap(d_labels_a, d_labels_b);
        if (changed == 0) {
            break;
        }
    }

    if (!check(cudaMemset(d_area, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_min_x, 0xFF, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_min_y, 0xFF, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_max_x, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_max_y, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_sum_x, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_sum_y, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_peak, 0, label_slots * sizeof(unsigned int))) ||
        !check(cudaMemset(d_active_count, 0, sizeof(unsigned int)))) {
        cleanup();
        return 11;
    }

    reduce_components_kernel<<<blocks, threads>>>(
        d_binary_active,
        d_labels_a,
        d_image,
        roi_width,
        roi_height,
        d_area,
        d_min_x,
        d_min_y,
        d_max_x,
        d_max_y,
        d_sum_x,
        d_sum_y,
        d_peak,
        d_active_labels,
        d_active_count);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cleanup();
        return 12;
    }

    unsigned int active_count = 0u;
    if (!check(cudaMemcpy(&active_count, d_active_count, sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
        cleanup();
        return 13;
    }
    if (active_count == 0u) {
        *detections_count_out = 0;
        cleanup();
        return 0;
    }

    if (!check(cudaMalloc(&d_detections, active_count * sizeof(RawDetectionCandidate))) ||
        !check(cudaMemset(d_detections_count, 0, sizeof(unsigned int)))) {
        cleanup();
        return 14;
    }

    const int gather_blocks = ceil_div(static_cast<int>(active_count), threads);
    gather_components_kernel<<<gather_blocks, threads>>>(
        d_active_labels,
        active_count,
        d_area,
        d_min_x,
        d_min_y,
        d_max_x,
        d_max_y,
        d_sum_x,
        d_sum_y,
        d_peak,
        d_threshold_map,
        threshold_constant,
        roi_width,
        roi_height,
        min_area,
        max_area,
        d_detections,
        d_detections_count);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
        cleanup();
        return 15;
    }

    unsigned int detections_count = 0u;
    if (!check(cudaMemcpy(&detections_count, d_detections_count, sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
        cleanup();
        return 16;
    }
    detections_count = min(detections_count, static_cast<unsigned int>(max_detections));
    if (detections_count > 0u &&
        !check(cudaMemcpy(detections_out, d_detections, detections_count * sizeof(RawDetectionCandidate), cudaMemcpyDeviceToHost))) {
        cleanup();
        return 17;
    }

    *detections_count_out = static_cast<int>(detections_count);
    cleanup();
    return 0;
}

extern "C" int otsu_cuda_postprocess_device_u8(
    void* cache_handle,
    uint16_t threshold_constant,
    int dilation,
    unsigned int min_area,
    unsigned int max_area,
    RawDetectionCandidate* detections_out,
    int max_detections,
    int* detections_count_out) {
    return postprocess_cached_impl(
        static_cast<OtsuCudaCache*>(cache_handle),
        threshold_constant,
        dilation,
        min_area,
        max_area,
        detections_out,
        max_detections,
        detections_count_out);
}

extern "C" int otsu_cuda_create_preprocess_cache(
    const uint8_t* roi_image_ptr,
    const uint8_t* binary_ptr,
    const uint16_t* threshold_map_ptr,
    int roi_width,
    int roi_height,
    void** cache_out) {
    if (!roi_image_ptr || !binary_ptr || !cache_out) {
        return 1;
    }

    auto* cache = new (std::nothrow) OtsuCudaCache();
    if (!cache) {
        return 2;
    }

    cache->roi_width = roi_width;
    cache->roi_height = roi_height;
    cache->roi_pixels = roi_width * roi_height;

    if (cache->roi_pixels > 0) {
        if (!check(cudaMalloc(&cache->image, cache->roi_pixels * sizeof(uint8_t))) ||
            !check(cudaMalloc(&cache->binary, cache->roi_pixels * sizeof(uint8_t)))) {
            destroy_cache(cache);
            return 3;
        }
        if (!check(cudaMemcpy(
                cache->image,
                roi_image_ptr,
                cache->roi_pixels * sizeof(uint8_t),
                cudaMemcpyHostToDevice)) ||
            !check(cudaMemcpy(
                cache->binary,
                binary_ptr,
                cache->roi_pixels * sizeof(uint8_t),
                cudaMemcpyHostToDevice))) {
            destroy_cache(cache);
            return 4;
        }
        if (threshold_map_ptr != nullptr) {
            if (!check(cudaMalloc(&cache->threshold_map, cache->roi_pixels * sizeof(uint16_t))) ||
                !check(cudaMemcpy(
                    cache->threshold_map,
                    threshold_map_ptr,
                    cache->roi_pixels * sizeof(uint16_t),
                    cudaMemcpyHostToDevice))) {
                destroy_cache(cache);
                return 5;
            }
        }
    }

    *cache_out = cache;
    return 0;
}

extern "C" void otsu_cuda_destroy_preprocess_cache(void* cache_handle) {
    destroy_cache(static_cast<OtsuCudaCache*>(cache_handle));
}
