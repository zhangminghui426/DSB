#include "dsb_infer/ops.hpp"

#include <cmath>

namespace dsb {

namespace {

// coordinate key: pack (x, y, z) with known grid (clamped to grid range).
int64_t key3(int x, int y, int z) {
  return ((int64_t)x << 40) | ((int64_t)y << 20) | (int64_t)z;
}

// per-layer view of the weight blob: (Cout, K, Cin), K == offsets.size().
struct LayerW {
  const SparseLayer &ly;
  const std::vector<float> &blob;
  size_t wbase;
  int K;
  const float *W() const { return blob.data() + ly.w_off; }
  float w(int co, int k, int ci) const {
    return blob[ly.w_off + ((size_t)co * K + k) * ly.in_c + ci];
  }
  const float *bn(int part, int co) const {
    // parts: 0 bias, 1 gamma, 2 beta, 3 mean, 4 var
    return blob.data() + ly.w_off + ly.w_len + (size_t)part * ly.b_len + co;
  }
};

}  // namespace

// Sparse encoder forward, CPU reference. Mirrors models/backbones/sparse_encoder.py
// forward with spconv semantics:
//   - coords are (batch, x, y, z); kernel/stride/pad in (x, y, z) order
//   - SubM conv (SparseBasicBlock): output voxels == input voxels, centerd
//     kernel offsets
//   - strided SparseConv: out = (in + pad - offset)/stride for divisible dims
//   - conv weights (Cout, K, Cin), K enumerated z->y->x (offsets from meta)
//   - final dense: sparse_shape (X, Y, Z) -> (C, X, Y, Z) -> fold Z
Tensor sparse_encoder_forward(const Config &cfg, const Tensor &feats,
                              const Tensor &coords, const Tensor &sizes,
                              const std::vector<float> &blob) {
  // ---- mean reduce: feats (M, 5) rows grouped by voxel order -> (M', 5)
  int nvox = (int)coords.shape[0];
  int feats_dim = (int)feats.shape[1];
  Tensor cur({nvox, feats_dim});
  {
    int r = 0;
    for (int v = 0; v < nvox; ++v) {
      int cnt = (int)std::lround(sizes.data[v]);
      for (int d = 0; d < feats_dim; ++d) {
        double s = 0;
        for (int i = 0; i < cnt; ++i) s += feats.data[(r + i) * feats_dim + d];
        cur.data[v * feats_dim + d] = (float)(s / std::max(cnt, 1));
      }
      r += cnt;
    }
  }

  // coords (M', 4) -> (x, y, z) int rows
  std::vector<std::vector<int>> in_coords(nvox);
  for (int v = 0; v < nvox; ++v) {
    in_coords[v] = {(int)std::lround(coords.data[v * 4]),
                    (int)std::lround(coords.data[v * 4 + 1]),
                    (int)std::lround(coords.data[v * 4 + 2])};
  }
  std::vector<int64_t> in_keys(nvox);
  for (int v = 0; v < nvox; ++v)
    in_keys[v] = key3(in_coords[v][0], in_coords[v][1], in_coords[v][2]);

  for (const auto &ly : cfg.sparse_layers) {
    int Cin = ly.in_c, Cout = ly.out_c;
    int K = (int)ly.offsets.size();
    // input lookup
    std::unordered_map<int64_t, int> in_map;
    for (int v = 0; v < (int)in_keys.size(); ++v) in_map[in_keys[v]] = v;

    if (ly.type == "subm") {
      // output voxels == input voxels
      Tensor nxt({(int)in_coords.size(), Cout});
      for (int v = 0; v < (int)in_coords.size(); ++v) {
        for (int co = 0; co < Cout; ++co) {
          float acc = blob[ly.w_off + ly.w_len + co];  // bias
          for (int k = 0; k < K; ++k) {
            int nk[3] = {in_coords[v][0] + ly.offsets[k][0],
                         in_coords[v][1] + ly.offsets[k][1],
                         in_coords[v][2] + ly.offsets[k][2]};
            auto it = in_map.find(key3(nk[0], nk[1], nk[2]));
            if (it == in_map.end()) continue;
            const float *xf = cur.data.data() + (size_t)it->second * Cin;
            const float *wf = blob.data() + ly.w_off + ((size_t)co * K + k) * Cin;
            for (int ci = 0; ci < Cin; ++ci) acc += wf[ci] * xf[ci];
          }
          // bn + relu
          float mean = blob[ly.w_off + ly.w_len + 3 * (size_t)ly.b_len + co];
          float var = blob[ly.w_off + ly.w_len + 4 * (size_t)ly.b_len + co];
          float g = blob[ly.w_off + ly.w_len + (size_t)ly.b_len + co];
          float b = blob[ly.w_off + ly.w_len + 2 * (size_t)ly.b_len + co];
          float v = (acc - mean) / std::sqrt(var + ly.bn_eps) * g + b;
          nxt.data[v * Cout + co] = std::max(v, 0.0f);
        }
      }
      cur = std::move(nxt);
    } else {
      // strided: collect output voxels + contributions
      std::unordered_map<int64_t, int> out_map;
      std::vector<std::vector<int>> out_coords;
      std::vector<std::vector<std::pair<int, int>>> contribs;
      for (int u = 0; u < (int)in_coords.size(); ++u) {
        for (int k = 0; k < K; ++k) {
          int oc[3];
          bool ok = true;
          for (int d = 0; d < 3; ++d) {
            int num = in_coords[u][d] + ly.pad[d] - ly.offsets[k][d];
            if (num % ly.stride[d] != 0) { ok = false; break; }
            oc[d] = num / ly.stride[d];
          }
          if (!ok) continue;
          int64_t key = key3(oc[0], oc[1], oc[2]);
          auto it = out_map.find(key);
          int oi;
          if (it == out_map.end()) {
            oi = (int)out_coords.size();
            out_map[key] = oi;
            out_coords.push_back({oc[0], oc[1], oc[2]});
            contribs.emplace_back();
          } else {
            oi = it->second;
          }
          contribs[oi].push_back({u, k});
        }
      }
      Tensor nxt({(int)out_coords.size(), Cout});
      for (int o = 0; o < (int)out_coords.size(); ++o) {
        for (int co = 0; co < Cout; ++co) {
          float acc = blob[ly.w_off + ly.w_len + co];
          for (auto &ck : contribs[o]) {
            const float *xf = cur.data.data() + (size_t)ck.first * Cin;
            const float *wf = blob.data() + ly.w_off +
                              ((size_t)co * K + ck.second) * Cin;
            for (int ci = 0; ci < Cin; ++ci) acc += wf[ci] * xf[ci];
          }
          float mean = blob[ly.w_off + ly.w_len + 3 * (size_t)ly.b_len + co];
          float var = blob[ly.w_off + ly.w_len + 4 * (size_t)ly.b_len + co];
          float g = blob[ly.w_off + ly.w_len + (size_t)ly.b_len + co];
          float b = blob[ly.w_off + ly.w_len + 2 * (size_t)ly.b_len + co];
          float v = (acc - mean) / std::sqrt(var + ly.bn_eps) * g + b;
          nxt.data[o * Cout + co] = std::max(v, 0.0f);
        }
      }
      cur = std::move(nxt);
      in_keys.clear();
      for (auto &c : out_coords) in_keys.push_back(key3(c[0], c[1], c[2]));
      in_coords = std::move(out_coords);
    }
  }

  // ---- dense: cur (M', C) + in_coords (M') -> (C, X, Y, Z) -> fold Z
  int X = cfg.sparse_out[0], Y = cfg.sparse_out[1], Z = cfg.sparse_out[2];
  int C = (int)cur.shape[1];
  std::unordered_map<int64_t, int> final_map;
  for (int v = 0; v < (int)in_coords.size(); ++v)
    final_map[key3(in_coords[v][0], in_coords[v][1], in_coords[v][2])] = v;

  // out (C*Z, X, Y): out[(cz*X + x)*Y + y] = feat[hash(x,y,z)][c]
  Tensor out({C * Z, X, Y});
  for (int x = 0; x < X; ++x)
    for (int y = 0; y < Y; ++y)
      for (int z = 0; z < Z; ++z) {
        auto it = final_map.find(key3(x, y, z));
        if (it == final_map.end()) continue;
        int v = it->second;
        for (int c = 0; c < C; ++c)
          out.data[(c * Z + z) * (X * Y) + x * Y + y] = cur.data[v * C + c];
      }
  return out;
}

}  // namespace dsb
