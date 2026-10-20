// 部署配置：与 3rdparty/deepleraing/deploy/config.py DeployConfig 对齐。
#include "dsb_infer/config.hpp"

#include <fstream>
#include <sstream>

namespace dsb {

namespace {

int to_int(const std::string &s) { return std::stoi(s); }
float to_float(const std::string &s) { return std::stof(s); }
bool to_bool(const std::string &s) { return s == "1" || s == "true"; }

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

}  // namespace

Config load_config(const std::string &path) {
  Config cfg;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    const auto p = line.find('=');
    if (line.empty() || line[0] == '#' || p == std::string::npos) continue;
    const auto k = line.substr(0, p);
    const auto v = line.substr(p + 1);
    if (k == "image_size") {
      const auto d = split(v, 'x');
      cfg.image_h = to_int(d.at(0));
      cfg.image_w = to_int(d.at(1));
    } else if (k == "num_camera") {
      cfg.num_camera = to_int(v);
    } else if (k == "num_depth") {
      cfg.num_depth = to_int(v);
    } else if (k == "feat_h") {
      cfg.feat_h = to_int(v);
    } else if (k == "feat_w") {
      cfg.feat_w = to_int(v);
    } else if (k == "camera_embed_dim") {
      cfg.camera_embed_dim = to_int(v);
    } else if (k == "downsample_bev_h") {
      cfg.downsample_bev_h = to_int(v);
    } else if (k == "downsample_bev_w") {
      cfg.downsample_bev_w = to_int(v);
    } else if (k == "bev_h") {
      cfg.bev_h = to_int(v);
    } else if (k == "bev_w") {
      cfg.bev_w = to_int(v);
    } else if (k == "bev_z") {
      cfg.bev_z = to_int(v);
    } else if (k == "bev_x_bound") {
      cfg.bev_x = to_float(v);
    } else if (k == "bev_y_bound") {
      cfg.bev_y = to_float(v);
    } else if (k == "bev_z_bound") {
      cfg.bev_z = to_float(v);
    } else if (k == "voxel_size") {
      cfg.voxel_size = to_float(v);
    } else if (k == "point_range") {
      const auto d = split(v, ',');
      cfg.point_range = {to_float(d.at(0)), to_float(d.at(1)),
                         to_float(d.at(2)), to_float(d.at(3)),
                         to_float(d.at(4)), to_float(d.at(5))};
    } else if (k == "point_cloud_range") {
      const auto d = split(v, ',');
      cfg.point_cloud_range = {to_float(d.at(0)), to_float(d.at(1)),
                               to_float(d.at(2)), to_float(d.at(3)),
                               to_float(d.at(4)), to_float(d.at(5))};
    } else if (k == "max_voxels") {
      cfg.max_voxels = to_int(v);
    } else if (k == "max_points_per_voxel") {
      cfg.max_points_per_voxel = to_int(v);
    } else if (k == "sparse_encoder_out_channels") {
      cfg.sparse_encoder_out_channels = to_int(v);
    } else if (k == "sparse_encoder_layers") {
      const auto d = split(v, ',');
      cfg.sparse_encoder_layers.clear();
      for (const auto &t : d) cfg.sparse_encoder_layers.push_back(to_int(t));
    } else if (k == "sparse_encoder_middle") {
      cfg.sparse_encoder_middle = to_int(v);
    } else if (k == "det_topk") {
      cfg.det_topk = to_int(v);
    } else if (k == "max_det") {
      cfg.max_det = to_int(v);
    } else if (k == "nms_iou_threshold") {
      cfg.nms_iou_threshold = to_float(v);
    } else if (k == "score_thr") {
      cfg.score_thr = to_float(v);
    }
  }
  return cfg;
}

}  // namespace dsb
