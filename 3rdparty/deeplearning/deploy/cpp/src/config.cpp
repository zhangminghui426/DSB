// 部署配置解析：读取 tools/export_cpp_input.py 产出的 meta.txt。
// 格式：key = 空格分隔值（'#' 注释），sparse / downsample 层表为结构化行。
#include "config.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace dsb {

namespace {

std::vector<std::string> split(const std::string &s, char sep = ' ') {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::vector<float> parse_floats(const std::string &v) {
  const auto t = split(v);
  std::vector<float> out;
  out.reserve(t.size());
  for (const auto &s : t) out.push_back(std::stof(s));
  return out;
}

std::vector<int> parse_ints(const std::string &v) {
  const auto t = split(v);
  std::vector<int> out;
  out.reserve(t.size());
  for (const auto &s : t) out.push_back(std::stoi(s));
  return out;
}

}  // namespace

Config load_config(const std::string &path) {
  Config cfg;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config: " + path);

  std::string line;
  while (std::getline(f, line)) {
    const auto p = line.find('=');
    if (line.empty() || line[0] == '#' || p == std::string::npos) continue;
    const auto k = line.substr(0, p);
    const auto v = line.substr(p + 1);
    auto trim = [](std::string s) {
      s.erase(0, s.find_first_not_of(" \t"));
      s.erase(s.find_last_not_of(" \t") + 1);
      return s;
    };
    const auto key = trim(k), val = trim(v);

    if (key == "B") cfg.B = std::stoi(val);
    else if (key == "N") cfg.N = std::stoi(val);
    else if (key == "image_h") cfg.image_h = std::stoi(val);
    else if (key == "image_w") cfg.image_w = std::stoi(val);
    else if (key == "feat_h") cfg.feat_h = std::stoi(val);
    else if (key == "feat_w") cfg.feat_w = std::stoi(val);
    else if (key == "xbound") cfg.xbound = parse_floats(val);
    else if (key == "ybound") cfg.ybound = parse_floats(val);
    else if (key == "zbound") cfg.zbound = parse_floats(val);
    else if (key == "dbound") cfg.dbound = parse_floats(val);
    else if (key == "nx") cfg.nx = parse_ints(val);
    else if (key == "lidar_pcr") cfg.lidar_pcr = parse_floats(val);
    else if (key == "lidar_voxel") cfg.lidar_voxel = parse_floats(val);
    else if (key == "lidar_grid") cfg.lidar_grid = parse_ints(val);
    else if (key == "lidar_max_pts") cfg.lidar_max_pts = std::stoi(val);
    else if (key == "lidar_max_voxels") cfg.lidar_max_voxels = std::stoi(val);
    else if (key == "sparse_out") cfg.sparse_out = parse_ints(val);
    else if (key == "vt_out_c") cfg.vt_out_c = std::stoi(val);
    else if (key == "head_nclass") cfg.head_nclass = std::stoi(val);
    else if (key == "head_nprop") cfg.head_nprop = std::stoi(val);
    else if (key == "head_nms_k") cfg.head_nms_k = std::stoi(val);
    else if (key == "head_bev") cfg.head_bev = std::stoi(val);
    else if (key == "head_pc") cfg.head_pc = parse_floats(val);
    else if (key == "head_osf") cfg.head_osf = std::stoi(val);
    else if (key == "head_voxel") cfg.head_voxel = parse_floats(val);
    else if (key == "head_post") cfg.head_post = parse_floats(val);
    else if (key == "head_score_thr") cfg.head_score_thr = std::stof(val);
    else if (key == "sparse_layer_N") {
      const int n = std::stoi(val);
      cfg.sparse_layers.reserve(n);
      for (int i = 0; i < n; ++i) {
        if (!std::getline(f, line)) break;
        const auto t = split(line);
        if (t.size() < 15) { --i; continue; }
        SparseLayer ly;
        ly.type = t[0];
        ly.in_c = std::stoi(t[1]);
        ly.out_c = std::stoi(t[2]);
        ly.kernel = {std::stoi(t[3]), std::stoi(t[4]), std::stoi(t[5])};
        ly.stride = {std::stoi(t[6]), std::stoi(t[7]), std::stoi(t[8])};
        ly.pad = {std::stoi(t[9]), std::stoi(t[10]), std::stoi(t[11])};
        const int noff = std::stoi(t[12]);
        ly.w_off = std::stoull(t[13]);
        ly.w_len = std::stoull(t[14]);
        ly.b_len = std::stoi(t[15]);
        ly.bn_eps = std::stof(t[16]);
        for (int j = 0; j < noff && std::getline(f, line); ++j)
          ly.offsets.push_back(parse_ints(line));
        cfg.sparse_layers.push_back(std::move(ly));
      }
    } else if (key == "downsample_N") {
      const int n = std::stoi(val);
      cfg.down_layers.reserve(n);
      for (int i = 0; i < n; ++i) {
        if (!std::getline(f, line)) break;
        const auto t = split(line);
        if (t.size() < 9) { --i; continue; }
        DownLayer dl;
        dl.in_c = std::stoi(t[0]);
        dl.out_c = std::stoi(t[1]);
        dl.k = std::stoi(t[2]);
        dl.stride = std::stoi(t[3]);
        dl.pad = std::stoi(t[4]);
        dl.w_off = std::stoull(t[5]);
        dl.w_len = std::stoull(t[6]);
        dl.b_len = std::stoi(t[7]);
        dl.bn_eps = std::stof(t[8]);
        cfg.down_layers.push_back(std::move(dl));
      }
    }
  }
  return cfg;
}

}  // namespace dsb
