// ROS2 wrapper around the CUDA-BEVFusion TensorRT core.
//
// Subscribes: 6 × sensor_msgs/Image (camera order: FRONT, FRONT_RIGHT,
// FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT), 1 × sensor_msgs/PointCloud2
// (x,y,z,intensity[,time], LiDAR frame), optional std_msgs/Float32MultiArray
// with 4×6×4×4 calibration matrices.
// Publishes: dsb_perception/BoundingBoxArray (gravity-center boxes) and
// visualization_msgs/MarkerArray.
//
// Synchronization: LiDAR frames drive inference. Each LiDAR stamp is matched
// to the nearest cached image per camera (within sync_slop), then the frame is
// pushed to a single-slot queue served by a worker thread. If inference is
// slower than the LiDAR period the older pending frame is dropped.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "bevfusion/bevfusion.hpp"
#include "dsb_perception/msg/bounding_box_array.hpp"

#ifndef DSB_DEPLOY_DIR_DEFAULT
#define DSB_DEPLOY_DIR_DEFAULT ""
#endif

namespace dsb_perception {

namespace {

constexpr int kNumCameras = 6;
constexpr int kImageChannels = 3;
constexpr int kMatrixFloats = kNumCameras * 16;

const std::array<const char *, 10> kClassNames = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"};

const std::array<std::array<float, 3>, 10> kClassColors = {{
    {1.0f, 0.62f, 0.0f},   {1.0f, 0.39f, 0.28f}, {0.91f, 0.59f, 0.27f},
    {1.0f, 0.27f, 0.0f},   {1.0f, 0.55f, 0.0f},  {0.44f, 0.50f, 0.56f},
    {1.0f, 0.24f, 0.39f},  {0.86f, 0.08f, 0.24f}, {0.0f, 0.0f, 0.90f},
    {0.18f, 0.31f, 0.31f},
}};

nvtype::half to_half(float value) {
  __half h = __float2half_rn(value);
  nvtype::half out;
  std::memcpy(&out, &h, sizeof(out));
  return out;
}

int field_offset(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name,
                 bool required) {
  for (const auto &f : msg.fields) {
    if (f.name != name) continue;
    if (f.datatype != sensor_msgs::msg::PointField::FLOAT32) {
      if (required) {
        throw std::runtime_error("PointCloud2 field '" + name + "' is not float32");
      }
      return -1;
    }
    return static_cast<int>(f.offset);
  }
  if (required) throw std::runtime_error("PointCloud2 missing field '" + name + "'");
  return -1;
}

int field_offset_f64(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name) {
  for (const auto &f : msg.fields) {
    if (f.name == name && f.datatype == sensor_msgs::msg::PointField::FLOAT64) {
      return static_cast<int>(f.offset);
    }
  }
  return -1;
}

}  // namespace

class BevfusionNode : public rclcpp::Node {
 public:
  BevfusionNode() : rclcpp::Node("bevfusion_node") {
    deploy_dir_ = declare_parameter<std::string>(
        "deploy_dir", std::string(DSB_DEPLOY_DIR_DEFAULT));
    if (deploy_dir_.empty()) {
      const char *env = std::getenv("DSB_DEPLOY_DIR");
      deploy_dir_ = (env != nullptr && env[0] != '\0')
                        ? std::string(env)
                        : std::string(DSB_DEPLOY_DIR_DEFAULT);
    }
    model_name_ = declare_parameter<std::string>("model_name", "resnet50");
    image_width_ = declare_parameter<int>("image_width", 1600);
    image_height_ = declare_parameter<int>("image_height", 900);
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/livox/lidar");
    frame_id_ = declare_parameter<std::string>("frame_id", "lidar");
    max_points_ = declare_parameter<int>("max_points", 300000);
    sync_slop_ = declare_parameter<double>("sync_slop", 0.05);
    max_camera_frames_ = declare_parameter<int>("max_camera_frames", 3);
    publish_markers_ = declare_parameter<bool>("publish_markers", true);

    const std::vector<std::string> default_cams = {
        "/camera/front/image_raw",      "/camera/front_right/image_raw",
        "/camera/front_left/image_raw", "/camera/back/image_raw",
        "/camera/back_left/image_raw",  "/camera/back_right/image_raw"};
    camera_topics_ = declare_parameter<std::vector<std::string>>(
        "camera_topics", default_cams);
    if (camera_topics_.size() != kNumCameras) {
      throw std::runtime_error("camera_topics must contain exactly 6 topics");
    }

    load_calibration_parameters();
    create_core();

    boxes_pub_ = create_publisher<dsb_perception::msg::BoundingBoxArray>(
        "~/boxes", rclcpp::QoS(10));
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/markers", rclcpp::QoS(10));

    for (int i = 0; i < kNumCameras; ++i) {
      image_subs_[i] = create_subscription<sensor_msgs::msg::Image>(
          camera_topics_[i], rclcpp::SensorDataQoS(),
          [this, i](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            image_callback(i, msg);
          });
    }
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          lidar_callback(msg);
        });
    matrices_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "~/matrices", rclcpp::QoS(10),
        [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
          matrices_callback(msg);
        });

    worker_ = std::thread([this]() { worker_loop(); });

    RCLCPP_INFO(get_logger(), "model: %s/model/%s", deploy_dir_.c_str(),
                model_name_.c_str());
    RCLCPP_INFO(get_logger(), "LiDAR: %s", lidar_topic_.c_str());
    for (int i = 0; i < kNumCameras; ++i) {
      RCLCPP_INFO(get_logger(), "camera[%d]: %s", i, camera_topics_[i].c_str());
    }
  }

  ~BevfusionNode() override {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

 private:
  struct ImageFrame {
    rclcpp::Time stamp;
    std::shared_ptr<std::vector<unsigned char>> data;
  };

  struct SyncedFrame {
    rclcpp::Time stamp;
    std::array<std::shared_ptr<std::vector<unsigned char>>, kNumCameras> images;
    std::vector<nvtype::half> points;
    int num_points = 0;
  };

  void load_calibration_parameters() {
    auto load = [&](const char *name) {
      auto values = declare_parameter<std::vector<double>>(name, std::vector<double>{});
      if (static_cast<int>(values.size()) != kMatrixFloats) {
        throw std::runtime_error(std::string(name) + " must have 96 values (6x4x4)");
      }
      std::array<float, kMatrixFloats> out{};
      for (int i = 0; i < kMatrixFloats; ++i) out[i] = static_cast<float>(values[i]);
      return out;
    };
    camera2lidar_ = load("camera2lidar");
    camera_intrinsics_ = load("camera_intrinsics");
    lidar2image_ = load("lidar2image");
    img_aug_matrix_ = load("img_aug_matrix");
  }

  void create_core() {
    if (dlopen("libcustom_layernorm.so", RTLD_NOW) == nullptr) {
      RCLCPP_WARN(get_logger(), "libcustom_layernorm.so not loaded: %s", dlerror());
    }
    const std::string model_dir = deploy_dir_ + "/model/" + model_name_;
    bevfusion::CoreParameter p;
    p.camera_model = model_dir + "/build/camera.backbone.plan";
    p.camera_vtransform = model_dir + "/build/camera.vtransform.plan";
    p.transfusion = model_dir + "/build/fuser.plan";
    p.transbbox.model = model_dir + "/build/head.bbox.fp32.plan";

    p.normalize.image_width = image_width_;
    p.normalize.image_height = image_height_;
    p.normalize.num_camera = kNumCameras;
    p.normalize.output_width = 704;
    p.normalize.output_height = 256;
    p.normalize.resize_lim = 0.48f;
    p.normalize.interpolation = bevfusion::camera::Interpolation::Bilinear;
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    p.normalize.method =
        bevfusion::camera::NormMethod::mean_std(mean, std, 1.0f / 255.0f, 0.0f);

    p.geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    p.geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
    p.geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
    p.geometry.dbound = nvtype::Float3(1.0f, 60.0f, 0.5f);
    p.geometry.geometry_dim = nvtype::Int3(360, 360, 80);
    p.geometry.feat_width = 88;
    p.geometry.feat_height = 32;
    p.geometry.image_width = 704;
    p.geometry.image_height = 256;
    p.geometry.num_camera = kNumCameras;

    p.lidar_scn.voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0f);
    p.lidar_scn.voxelization.max_range = nvtype::Float3(54.0f, 54.0f, 3.0f);
    p.lidar_scn.voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
    p.lidar_scn.voxelization.grid_size =
        bevfusion::lidar::VoxelizationParameter::compute_grid_size(
            p.lidar_scn.voxelization.max_range, p.lidar_scn.voxelization.min_range,
            p.lidar_scn.voxelization.voxel_size);
    p.lidar_scn.voxelization.max_points_per_voxel = 10;
    p.lidar_scn.voxelization.max_points = max_points_;
    p.lidar_scn.voxelization.max_voxels = 160000;
    p.lidar_scn.voxelization.num_feature = 5;
    p.lidar_scn.model = model_dir + "/lidar.backbone.xyz.onnx";
    p.lidar_scn.precision = bevfusion::lidar::Precision::Float16;
    p.lidar_scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

    p.transbbox.out_size_factor = 8;
    p.transbbox.voxel_size = {0.075f, 0.075f};
    p.transbbox.pc_range = {-54.0f, -54.0f};
    p.transbbox.post_center_range_start = {-61.2f, -61.2f, -10.0f};
    p.transbbox.post_center_range_end = {61.2f, 61.2f, 10.0f};
    p.transbbox.confidence_threshold = 0.12f;
    p.transbbox.sorted_bboxes = true;

    core_ = bevfusion::create_core(p);
    if (core_ == nullptr) {
      throw std::runtime_error("failed to create BEVFusion core (check plans in " +
                               model_dir + "/build)");
    }
    core_->set_timer(false);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed");
    }
    core_->update(camera2lidar_.data(), camera_intrinsics_.data(),
                  lidar2image_.data(), img_aug_matrix_.data(), stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      throw std::runtime_error("cudaStreamSynchronize failed");
    }
  }

  void image_callback(int index, const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (static_cast<int>(msg->width) != image_width_ ||
        static_cast<int>(msg->height) != image_height_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "camera[%d] image %ux%u != expected %dx%d", index,
                           msg->width, msg->height, image_width_, image_height_);
      return;
    }
    auto buf = std::make_shared<std::vector<unsigned char>>(
        static_cast<size_t>(image_width_) * image_height_ * kImageChannels);
    const auto &src = msg->data;
    const size_t row_bytes = static_cast<size_t>(image_width_) * kImageChannels;
    if (msg->encoding == "rgb8" || msg->encoding == "bgr8") {
      const bool bgr = msg->encoding == "bgr8";
      for (int y = 0; y < image_height_; ++y) {
        const unsigned char *row = src.data() + static_cast<size_t>(y) * msg->step;
        unsigned char *out = buf->data() + static_cast<size_t>(y) * row_bytes;
        if (!bgr) {
          std::memcpy(out, row, row_bytes);
        } else {
          for (int x = 0; x < image_width_; ++x) {
            out[x * 3 + 0] = row[x * 3 + 2];
            out[x * 3 + 1] = row[x * 3 + 1];
            out[x * 3 + 2] = row[x * 3 + 0];
          }
        }
      }
    } else if (msg->encoding == "rgba8" || msg->encoding == "bgra8") {
      const bool bgr = msg->encoding == "bgra8";
      for (int y = 0; y < image_height_; ++y) {
        const unsigned char *row = src.data() + static_cast<size_t>(y) * msg->step;
        unsigned char *out = buf->data() + static_cast<size_t>(y) * row_bytes;
        for (int x = 0; x < image_width_; ++x) {
          out[x * 3 + 0] = row[x * 4 + (bgr ? 2 : 0)];
          out[x * 3 + 1] = row[x * 4 + 1];
          out[x * 3 + 2] = row[x * 4 + (bgr ? 0 : 2)];
        }
      }
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "camera[%d] unsupported encoding '%s'", index,
                           msg->encoding.c_str());
      return;
    }

    ImageFrame frame;
    frame.stamp = rclcpp::Time(msg->header.stamp);
    frame.data = buf;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto &cache = image_cache_[index];
    cache.push_back(std::move(frame));
    while (static_cast<int>(cache.size()) > max_camera_frames_) cache.pop_front();
  }

  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    try {
      const int ox = field_offset(*msg, "x", true);
      const int oy = field_offset(*msg, "y", true);
      const int oz = field_offset(*msg, "z", true);
      const int oi = field_offset(*msg, "intensity", false);
      const int ot32 = field_offset(*msg, "time", false);
      const int ot64 = field_offset_f64(*msg, "timestamp");

      const size_t total = static_cast<size_t>(msg->width) * msg->height;
      const size_t cap = std::min<size_t>(total, static_cast<size_t>(max_points_));
      std::vector<nvtype::half> points(cap * 5, nvtype::half{});
      size_t n = 0;
      for (size_t k = 0; k < total && n < cap; ++k) {
        const auto *base = msg->data.data() + k * msg->point_step;
        const float x = *reinterpret_cast<const float *>(base + ox);
        const float y = *reinterpret_cast<const float *>(base + oy);
        const float z = *reinterpret_cast<const float *>(base + oz);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        const float intensity =
            oi >= 0 ? *reinterpret_cast<const float *>(base + oi) : 0.0f;
        float time = 0.0f;
        if (ot32 >= 0) {
          time = *reinterpret_cast<const float *>(base + ot32);
        } else if (ot64 >= 0) {
          // Livox PointCloud2: timestamp is offset_time in nanoseconds
          time = static_cast<float>(*reinterpret_cast<const double *>(base + ot64) * 1e-9);
        }
        points[n * 5 + 0] = to_half(x);
        points[n * 5 + 1] = to_half(y);
        points[n * 5 + 2] = to_half(z);
        points[n * 5 + 3] = to_half(intensity);
        points[n * 5 + 4] = to_half(time);
        ++n;
      }
      if (n == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "LiDAR cloud has no valid points");
        return;
      }
      points.resize(n * 5);

      const rclcpp::Time lidar_stamp(msg->header.stamp);
      SyncedFrame synced;
      synced.stamp = lidar_stamp;
      synced.points = std::move(points);
      synced.num_points = static_cast<int>(n);

      std::lock_guard<std::mutex> lock(cache_mutex_);
      for (int i = 0; i < kNumCameras; ++i) {
        const auto &cache = image_cache_[i];
        if (cache.empty()) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                               "sync: camera[%d] has no frame, drop LiDAR frame", i);
          return;
        }
        if (lidar_stamp.nanoseconds() <= 0) {
          synced.images[i] = cache.back().data;
          continue;
        }
        auto best = cache.begin();
        double best_diff = std::numeric_limits<double>::max();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
          double diff = std::abs((it->stamp - lidar_stamp).seconds());
          if (diff < best_diff) {
            best_diff = diff;
            best = it;
          }
        }
        if (best_diff > sync_slop_) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                               "sync: camera[%d] nearest frame %.1f ms away (>%.1f), drop",
                               i, best_diff * 1000.0, sync_slop_ * 1000.0);
          return;
        }
        synced.images[i] = best->data;
      }

      {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        if (pending_) {
          ++dropped_frames_;
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "inference busy, drop pending frame (%ld dropped)",
                               dropped_frames_);
        }
        pending_ = std::move(synced);
      }
      queue_cv_.notify_one();
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "LiDAR: %s", e.what());
    }
  }

  void matrices_callback(const std_msgs::msg::Float32MultiArray::ConstSharedPtr &msg) {
    constexpr size_t expected = 4 * kMatrixFloats;
    if (msg->data.size() != expected) {
      RCLCPP_WARN(get_logger(), "matrices must have %zu values, got %zu", expected,
                  msg->data.size());
      return;
    }
    std::lock_guard<std::mutex> lock(core_mutex_);
    std::copy_n(msg->data.begin(), kMatrixFloats, camera2lidar_.begin());
    std::copy_n(msg->data.begin() + kMatrixFloats, kMatrixFloats,
                camera_intrinsics_.begin());
    std::copy_n(msg->data.begin() + 2 * kMatrixFloats, kMatrixFloats,
                lidar2image_.begin());
    std::copy_n(msg->data.begin() + 3 * kMatrixFloats, kMatrixFloats,
                img_aug_matrix_.begin());
    core_->update(camera2lidar_.data(), camera_intrinsics_.data(),
                  lidar2image_.data(), img_aug_matrix_.data(), stream_);
    RCLCPP_INFO(get_logger(), "calibration matrices updated");
  }

  void worker_loop() {
    while (true) {
      SyncedFrame frame;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() { return stop_ || pending_.has_value(); });
        if (stop_ && !pending_) return;
        frame = std::move(*pending_);
        pending_.reset();
      }

      std::array<const unsigned char *, kNumCameras> images{};
      for (int i = 0; i < kNumCameras; ++i) images[i] = frame.images[i]->data();

      std::vector<bevfusion::head::transbbox::BoundingBox> boxes;
      {
        std::lock_guard<std::mutex> lock(core_mutex_);
        boxes = core_->forward(images.data(), frame.points.data(), frame.num_points,
                               stream_);
        cudaStreamSynchronize(stream_);
      }
      publish(boxes, frame.stamp);
      ++frame_count_;
      if (frame_count_ % 10 == 1) {
        RCLCPP_INFO(get_logger(), "frame %ld: %zu boxes", frame_count_, boxes.size());
      }
    }
  }

  void publish(const std::vector<bevfusion::head::transbbox::BoundingBox> &boxes,
               const rclcpp::Time &stamp) {
    dsb_perception::msg::BoundingBoxArray out;
    out.header.stamp = stamp;
    out.header.frame_id = frame_id_;
    out.boxes.reserve(boxes.size());
    for (const auto &b : boxes) {
      dsb_perception::msg::BoundingBox box;
      const int id = std::clamp(b.id, 0, static_cast<int>(kClassNames.size()) - 1);
      box.label = kClassNames[id];
      box.label_id = static_cast<uint8_t>(id);
      box.score = b.score;
      box.x = b.position.x;
      box.y = b.position.y;
      box.z = b.position.z;
      box.w = b.size.w;
      box.l = b.size.l;
      box.h = b.size.h;
      box.yaw = b.z_rotation;
      box.vx = b.velocity.vx;
      box.vy = b.velocity.vy;
      out.boxes.push_back(box);
    }
    boxes_pub_->publish(out);

    if (!publish_markers_) return;
    visualization_msgs::msg::MarkerArray markers;
    for (size_t i = 0; i < out.boxes.size(); ++i) {
      const auto &b = out.boxes[i];
      visualization_msgs::msg::Marker m;
      m.header = out.header;
      m.ns = "bevfusion";
      m.id = static_cast<int>(i);
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = b.x;
      m.pose.position.y = b.y;
      m.pose.position.z = b.z;
      m.pose.orientation.z = std::sin(b.yaw * 0.5f);
      m.pose.orientation.w = std::cos(b.yaw * 0.5f);
      m.scale.x = b.w;
      m.scale.y = b.l;
      m.scale.z = b.h;
      const auto &c = kClassColors[std::clamp(
          static_cast<int>(b.label_id), 0,
          static_cast<int>(kClassColors.size()) - 1)];
      m.color.r = c[0];
      m.color.g = c[1];
      m.color.b = c[2];
      m.color.a = 0.55f;
      m.lifetime = rclcpp::Duration::from_seconds(0.2);
      markers.markers.push_back(m);
    }
    markers_pub_->publish(markers);
  }

  std::string deploy_dir_;
  std::string model_name_;
  int image_width_ = 1600;
  int image_height_ = 900;
  int max_points_ = 300000;
  double sync_slop_ = 0.05;
  int max_camera_frames_ = 3;
  bool publish_markers_ = true;
  std::string lidar_topic_;
  std::string frame_id_;
  std::vector<std::string> camera_topics_;

  std::array<float, kMatrixFloats> camera2lidar_{};
  std::array<float, kMatrixFloats> camera_intrinsics_{};
  std::array<float, kMatrixFloats> lidar2image_{};
  std::array<float, kMatrixFloats> img_aug_matrix_{};

  std::mutex cache_mutex_;
  std::array<std::deque<ImageFrame>, kNumCameras> image_cache_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::optional<SyncedFrame> pending_;
  bool stop_ = false;
  long dropped_frames_ = 0;

  std::mutex core_mutex_;
  std::shared_ptr<bevfusion::Core> core_;
  cudaStream_t stream_ = nullptr;
  long frame_count_ = 0;
  std::thread worker_;

  std::array<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr, kNumCameras>
      image_subs_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr matrices_sub_;
  rclcpp::Publisher<dsb_perception::msg::BoundingBoxArray>::SharedPtr boxes_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
};

}  // namespace dsb_perception

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<dsb_perception::BevfusionNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("bevfusion_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
