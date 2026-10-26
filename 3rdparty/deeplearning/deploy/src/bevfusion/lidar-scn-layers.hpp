/*
 * Per-layer SCN execution helpers.
 *
 * libspconv's monolithic SCN graph miscomputes chained SparseConvolution
 * nodes on sparse neighborhoods; running one conv per engine matches torch
 * (lidar_bev corr ~0.99996). These kernels apply the ReLU/Add between
 * per-layer engines and fold the final sparse features into BEV.
 */
#ifndef __LIDAR_SCN_LAYERS_HPP__
#define __LIDAR_SCN_LAYERS_HPP__

#include <cstddef>

#include "common/dtype.hpp"

namespace bevfusion {
namespace lidar {

void scn_relu_half(nvtype::half* data, size_t n, void* stream);
void scn_add_half(nvtype::half* dst, const nvtype::half* src, size_t n, void* stream);
// feat (M, C), idx (M, 4) = [batch, x, y, z] -> out (C*Z, X, Y)
void scn_scatter_fold(const nvtype::half* feat, const int* idx, nvtype::half* out,
                      int M, int C, int X, int Y, int Z, void* stream);

};  // namespace lidar
};  // namespace bevfusion

#endif  // __LIDAR_SCN_LAYERS_HPP__
