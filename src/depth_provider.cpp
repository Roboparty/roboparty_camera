// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2025-2026 Luo1imasi

#include "depth_provider.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

size_t tensor_element_count(std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t& dim : shape) {
        if (dim == -1) {
            dim = 1;
        }
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::string resolve_asset_path(const std::string& base_dir, const std::string& asset_name) {
    const std::filesystem::path asset_path(asset_name);
    if (asset_path.is_absolute()) {
        return asset_path.string();
    }
    return (std::filesystem::path(base_dir) / asset_path).lexically_normal().string();
}

}  // namespace

class DepthProvider::Impl {
   public:
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::Value> input_tensor;
        std::unique_ptr<Ort::Value> output_tensor;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char*> input_names_raw;
        std::vector<const char*> output_names_raw;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<float> input_buffer;
        std::vector<float> output_buffer;
        size_t num_inputs = 0;
        size_t num_outputs = 0;
    };

    Impl(rclcpp::Node& node, Ort::Env& env, std::size_t expected_output_size) : node_(node), env_(env) {
        load_config();
        validate_config();
        if (static_cast<size_t>(output_dim()) != expected_output_size) {
            throw std::runtime_error("Depth output size does not match the policy perception field");
        }

        if (use_depth_encoder_) {
            setup_model();
        }
        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        depth_subscription_ = node_.create_subscription<sensor_msgs::msg::Image>(
            depth_topic_, data_qos, std::bind(&Impl::depth_image_callback, this, std::placeholders::_1));

        if (debug_vis_) {
            auto vis_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
            std::string topic_base = debug_depth_vis_topic_;
            if (!topic_base.empty() && topic_base.back() == '/') {
                topic_base.pop_back();
            }
            if (topic_base.empty()) {
                topic_base = "/debug_depth_vis";
            }
            debug_downsample_publisher_ = node_.create_publisher<sensor_msgs::msg::Image>(topic_base + "/downsample", vis_qos);
            debug_crop_publisher_ = node_.create_publisher<sensor_msgs::msg::Image>(topic_base + "/crop", vis_qos);
        }

        RCLCPP_INFO(node_.get_logger(), "depth_topic: %s", depth_topic_.c_str());
        RCLCPP_INFO(node_.get_logger(), "depth_encoder_model_path: %s", depth_encoder_model_path_.c_str());
        RCLCPP_INFO(node_.get_logger(), "depth output dim: %d", output_dim());
        RCLCPP_INFO(node_.get_logger(),
                    "depth crop inpaint: distance=%.3f m radius=%.1f, gaussian: ksize=%d sigma=%.3f",
                    depth_inpaint_distance_, depth_inpaint_radius_,
                    depth_gaussian_kernel_size_, depth_gaussian_sigma_);
        RCLCPP_INFO(node_.get_logger(), "depth history/encode ready (pulled by depth_node)");
    }

   private:
    void load_config() {
        auto declare_if_needed = [this](const std::string& name, const auto& default_value) {
            if (!node_.has_parameter(name)) {
                node_.declare_parameter(name, default_value);
            }
        };
        declare_if_needed("depth_topic", std::string("/camera/camera/depth/image_rect_raw"));
        declare_if_needed("depth_encoder_model_name", std::string("encoder.onnx"));
        declare_if_needed("use_depth_encoder", true);
        declare_if_needed("debug_vis", false);
        declare_if_needed("debug_depth_vis_topic", std::string("/debug_depth_vis"));
        declare_if_needed("depth_encoder_output_dim", 128);
        declare_if_needed("depth_input_width", 640);
        declare_if_needed("depth_input_height", 360);
        declare_if_needed("depth_width", 64);
        declare_if_needed("depth_height", 36);
        declare_if_needed("depth_crop_up", 18);
        declare_if_needed("depth_crop_down", 0);
        declare_if_needed("depth_crop_left", 16);
        declare_if_needed("depth_crop_right", 16);
        declare_if_needed("depth_history_taps",
                          std::vector<int64_t>{35, 30, 25, 20, 15, 10, 5, 0});
        declare_if_needed("depth_min_distance", 0.0f);
        declare_if_needed("depth_max_distance", 2.5f);
        declare_if_needed("depth_inpaint_distance", 0.2f);
        declare_if_needed("depth_inpaint_radius", 3.0);
        declare_if_needed("depth_gaussian_kernel_size", 3);
        declare_if_needed("depth_gaussian_sigma", 1.0);
        declare_if_needed("model_dir", std::string(""));

        node_.get_parameter("depth_topic", depth_topic_);
        node_.get_parameter("model_dir", model_dir_);
        node_.get_parameter("depth_encoder_model_name", depth_encoder_model_name_);
        node_.get_parameter("use_depth_encoder", use_depth_encoder_);
        node_.get_parameter("debug_vis", debug_vis_);
        node_.get_parameter("debug_depth_vis_topic", debug_depth_vis_topic_);
        node_.get_parameter("depth_encoder_output_dim", depth_encoder_output_dim_);
        node_.get_parameter("depth_input_width", depth_input_width_);
        node_.get_parameter("depth_input_height", depth_input_height_);
        node_.get_parameter("depth_width", depth_width_);
        node_.get_parameter("depth_height", depth_height_);
        node_.get_parameter("depth_crop_up", depth_crop_up_);
        node_.get_parameter("depth_crop_down", depth_crop_down_);
        node_.get_parameter("depth_crop_left", depth_crop_left_);
        node_.get_parameter("depth_crop_right", depth_crop_right_);
        node_.get_parameter("depth_history_taps", depth_history_taps_);
        node_.get_parameter("depth_min_distance", depth_min_distance_);
        node_.get_parameter("depth_max_distance", depth_max_distance_);
        node_.get_parameter("depth_inpaint_distance", depth_inpaint_distance_);
        node_.get_parameter("depth_inpaint_radius", depth_inpaint_radius_);
        node_.get_parameter("depth_gaussian_kernel_size", depth_gaussian_kernel_size_);
        node_.get_parameter("depth_gaussian_sigma", depth_gaussian_sigma_);

        depth_obs_width_ = depth_width_ - depth_crop_left_ - depth_crop_right_;
        depth_obs_height_ = depth_height_ - depth_crop_up_ - depth_crop_down_;
        depth_obs_num_ = depth_obs_width_ * depth_obs_height_;
        depth_encoder_model_path_ = resolve_asset_path(model_dir_, depth_encoder_model_name_);
    }

    void validate_config() const {

        if (depth_width_ <= 0 || depth_height_ <= 0 || depth_obs_width_ <= 0 || depth_obs_height_ <= 0) {
            throw std::runtime_error("Invalid depth downsample/crop dimensions");
        }
        if (depth_history_taps_.empty()) {
            throw std::runtime_error("depth_history_taps must not be empty");
        }
        for (size_t i = 0; i < depth_history_taps_.size(); ++i) {
            if (depth_history_taps_[i] < 0 ||
                (i > 0 && depth_history_taps_[i - 1] <= depth_history_taps_[i])) {
                throw std::runtime_error("depth_history_taps must be non-negative and strictly descending");
            }
        }
        if (depth_max_distance_ <= depth_min_distance_) {
            throw std::runtime_error("depth_max_distance must be greater than depth_min_distance");
        }
        if (depth_inpaint_distance_ < 0.0f) {
            throw std::runtime_error("depth_inpaint_distance must be non-negative");
        }
        if (depth_inpaint_radius_ < 0.0) {
            throw std::runtime_error("depth_inpaint_radius must be non-negative");
        }
        if (depth_gaussian_kernel_size_ != 0 &&
            (depth_gaussian_kernel_size_ < 3 || depth_gaussian_kernel_size_ % 2 == 0)) {
            throw std::runtime_error("depth_gaussian_kernel_size must be 0 or an odd integer >= 3");
        }
        if (depth_gaussian_sigma_ < 0.0) {
            throw std::runtime_error("depth_gaussian_sigma must be non-negative");
        }
        if (use_depth_encoder_ && depth_encoder_output_dim_ <= 0) {
            throw std::runtime_error("depth_encoder_output_dim must be positive when use_depth_encoder is true");
        }
    }

    int output_dim() const {
        return use_depth_encoder_
                   ? depth_encoder_output_dim_
                   : depth_obs_num_ * static_cast<int>(depth_history_taps_.size());
    }

    void setup_model() {
        Ort::SessionOptions session_options;
        session_options.DisablePerSessionThreads();
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        encoder_ctx_.session = std::make_unique<Ort::Session>(env_, depth_encoder_model_path_.c_str(), session_options);
        encoder_ctx_.num_inputs = encoder_ctx_.session->GetInputCount();
        if (encoder_ctx_.num_inputs != 1) {
            throw std::runtime_error("Only single-input depth encoder models are supported: " + depth_encoder_model_path_);
        }
        encoder_ctx_.input_names.resize(encoder_ctx_.num_inputs);
        for (size_t i = 0; i < encoder_ctx_.num_inputs; i++) {
            Ort::AllocatedStringPtr input_name = encoder_ctx_.session->GetInputNameAllocated(i, allocator_);
            encoder_ctx_.input_names[i] = input_name.get();
            auto type_info = encoder_ctx_.session->GetInputTypeInfo(i);
            encoder_ctx_.input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        }

        const size_t model_input_size = tensor_element_count(encoder_ctx_.input_shape);
        const size_t expected_input_size =
            static_cast<size_t>(depth_obs_num_) * depth_history_taps_.size();
        if (model_input_size != expected_input_size) {
            throw std::runtime_error("Depth encoder input size mismatch: model expects " +
                                     std::to_string(model_input_size) + " values, config provides " +
                                     std::to_string(expected_input_size));
        }
        encoder_ctx_.input_buffer.assign(expected_input_size, 0.0f);

        encoder_ctx_.num_outputs = encoder_ctx_.session->GetOutputCount();
        if (encoder_ctx_.num_outputs != 1) {
            throw std::runtime_error("Only single-output depth encoder models are supported: " + depth_encoder_model_path_);
        }
        encoder_ctx_.output_names.resize(encoder_ctx_.num_outputs);
        for (size_t i = 0; i < encoder_ctx_.num_outputs; i++) {
            Ort::AllocatedStringPtr output_name = encoder_ctx_.session->GetOutputNameAllocated(i, allocator_);
            encoder_ctx_.output_names[i] = output_name.get();
            auto type_info = encoder_ctx_.session->GetOutputTypeInfo(i);
            encoder_ctx_.output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        }

        const size_t model_output_size = tensor_element_count(encoder_ctx_.output_shape);
        if (model_output_size != static_cast<size_t>(depth_encoder_output_dim_)) {
            throw std::runtime_error("Depth encoder output size mismatch: model expects " +
                                     std::to_string(model_output_size) + " values, config provides " +
                                     std::to_string(depth_encoder_output_dim_));
        }
        encoder_ctx_.output_buffer.assign(static_cast<size_t>(depth_encoder_output_dim_), 0.0f);

        encoder_ctx_.input_names_raw.resize(encoder_ctx_.num_inputs);
        encoder_ctx_.output_names_raw.resize(encoder_ctx_.num_outputs);
        for (size_t i = 0; i < encoder_ctx_.num_inputs; i++) {
            encoder_ctx_.input_names_raw[i] = encoder_ctx_.input_names[i].c_str();
        }
        for (size_t i = 0; i < encoder_ctx_.num_outputs; i++) {
            encoder_ctx_.output_names_raw[i] = encoder_ctx_.output_names[i].c_str();
        }

        encoder_ctx_.memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        encoder_ctx_.input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
            *encoder_ctx_.memory_info, encoder_ctx_.input_buffer.data(), encoder_ctx_.input_buffer.size(),
            encoder_ctx_.input_shape.data(), encoder_ctx_.input_shape.size()));
        encoder_ctx_.output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
            *encoder_ctx_.memory_info, encoder_ctx_.output_buffer.data(), encoder_ctx_.output_buffer.size(),
            encoder_ctx_.output_shape.data(), encoder_ctx_.output_shape.size()));
    }

    // Convert ROS depth image to meters (CV_32F). Invalid / non-positive -> 0.
    // Range clipping happens after inpainting and Gaussian blur during normalization.
    cv::Mat depth_image_to_meters(const sensor_msgs::msg::Image& msg, bool is_u16) const {
        const int src_w = static_cast<int>(msg.width);
        const int src_h = static_cast<int>(msg.height);
        // OpenCV Mat header over message buffer (no copy of raw bytes until convert/clone).
        cv::Mat depth_m;
        if (is_u16) {
            const cv::Mat depth_mm(src_h, src_w, CV_16UC1,
                                  const_cast<uint8_t*>(msg.data.data()), msg.step);
            depth_mm.convertTo(depth_m, CV_32F, 0.001);  // mm -> m; 0 stays 0
        } else {
            const cv::Mat depth_f(src_h, src_w, CV_32FC1,
                                  const_cast<uint8_t*>(msg.data.data()), msg.step);
            depth_f.convertTo(depth_m, CV_32F);  // clone into owned buffer
            cv::patchNaNs(depth_m, 0.0);
        }

        cv::Mat invalid_mask;
        cv::compare(depth_m, 0.0f, invalid_mask, cv::CMP_LE);
        cv::Mat infinity_mask;
        cv::compare(depth_m, std::numeric_limits<float>::max(), infinity_mask, cv::CMP_GT);
        cv::bitwise_or(invalid_mask, infinity_mask, invalid_mask);
        depth_m.setTo(0.0f, invalid_mask);
        return depth_m;
    }

    std::vector<float> normalized_from_depth_m(const cv::Mat& depth_m) const {
        const float depth_range = depth_max_distance_ - depth_min_distance_;
        cv::Mat normalized;
        if (depth_range <= 0.0f) {
            normalized = cv::Mat::zeros(depth_m.size(), CV_32F);
        } else {
            normalized = (depth_m - depth_min_distance_) * (1.0f / depth_range);
            cv::min(normalized, 1.0f, normalized);
            cv::max(normalized, 0.0f, normalized);
        }
        if (!normalized.isContinuous()) {
            normalized = normalized.clone();
        }
        const float* ptr = normalized.ptr<float>(0);
        return std::vector<float>(ptr, ptr + static_cast<size_t>(normalized.rows * normalized.cols));
    }

    // Match the deployed parkour preprocessing: fill invalid/too-close pixels,
    // then smooth the cropped metric depth image before normalization.
    void inpaint_and_blur(cv::Mat& depth_m) const {
        if (depth_m.empty() || depth_m.type() != CV_32F) {
            return;
        }
        if (depth_inpaint_radius_ > 0.0) {
            cv::Mat hole_mask;
            cv::compare(depth_m, depth_inpaint_distance_, hole_mask, cv::CMP_LT);
            const int hole_count = cv::countNonZero(hole_mask);
            if (hole_count > 0 && hole_count < static_cast<int>(depth_m.total())) {
                cv::Mat inpainted;
                cv::inpaint(depth_m, hole_mask, inpainted, depth_inpaint_radius_, cv::INPAINT_NS);
                depth_m = inpainted;
            }
        }
        if (depth_gaussian_kernel_size_ >= 3) {
            cv::GaussianBlur(depth_m, depth_m,
                             cv::Size(depth_gaussian_kernel_size_, depth_gaussian_kernel_size_),
                             depth_gaussian_sigma_, depth_gaussian_sigma_);
        }
    }

    // ---- Preprocess ----
    std::vector<float> preprocess_depth_image(const sensor_msgs::msg::Image& msg) {
        const bool is_u16 = (msg.encoding == "16UC1");
        const bool is_f32 = (msg.encoding == "32FC1");
        if (!is_u16 && !is_f32) {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
                                 "Unsupported depth encoding: %s (expected 16UC1 or 32FC1)", msg.encoding.c_str());
            return {};
        }
        if (msg.width == 0 || msg.height == 0 || msg.data.empty()) {
            return {};
        }

        const int src_w = static_cast<int>(msg.width);
        const int src_h = static_cast<int>(msg.height);
        if (src_w != depth_input_width_ || src_h != depth_input_height_) {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
                                 "Depth input resolution mismatch, got %dx%d expected %dx%d",
                                 src_w, src_h, depth_input_width_, depth_input_height_);
        }

        const size_t bytes_per_pixel = is_u16 ? sizeof(uint16_t) : sizeof(float);
        const size_t min_step = static_cast<size_t>(src_w) * bytes_per_pixel;
        if (msg.step < min_step) {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
                                 "Invalid depth step: %u, expected at least %zu", msg.step, min_step);
            return {};
        }

        const cv::Mat depth_m = depth_image_to_meters(msg, is_u16);

        cv::Mat grid_m;
        // Match the instinct_onboard ParkourAgent depth preprocessing.
        cv::resize(depth_m, grid_m, cv::Size(depth_width_, depth_height_), 0, 0, cv::INTER_NEAREST);

        // Downsample debug: publish meters (no full-grid normalization).
        publish_debug_image(grid_m, msg, debug_downsample_publisher_);

        const cv::Rect crop_rect(depth_crop_left_, depth_crop_up_, depth_obs_width_, depth_obs_height_);
        if (crop_rect.x + crop_rect.width > depth_width_ || crop_rect.y + crop_rect.height > depth_height_) {
            RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 2000,
                                 "Invalid depth crop region on downsample grid: %dx%d+%dx%d in %dx%d grid",
                                 crop_rect.x, crop_rect.y, crop_rect.width, crop_rect.height, depth_width_,
                                 depth_height_);
            return {};
        }

        cv::Mat cropped_m = grid_m(crop_rect).clone();
        inpaint_and_blur(cropped_m);
        // Encoder path always needs crop normalization to [0, 1].
        std::vector<float> output = normalized_from_depth_m(cropped_m);
        publish_debug_image(output, depth_obs_width_, depth_obs_height_, msg, debug_crop_publisher_);
        return output;
    }

    void publish_debug_image(const cv::Mat& image_f32,
                             const sensor_msgs::msg::Image& source_msg,
                             const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
        if (!debug_vis_ || !publisher || image_f32.empty() || image_f32.type() != CV_32F) {
            return;
        }
        cv::Mat continuous = image_f32.isContinuous() ? image_f32 : image_f32.clone();
        sensor_msgs::msg::Image debug_msg;
        debug_msg.header = source_msg.header;
        debug_msg.height = static_cast<uint32_t>(continuous.rows);
        debug_msg.width = static_cast<uint32_t>(continuous.cols);
        debug_msg.encoding = "32FC1";
        debug_msg.is_bigendian = false;
        debug_msg.step = static_cast<uint32_t>(continuous.cols * sizeof(float));
        const size_t nbytes = static_cast<size_t>(continuous.rows) * debug_msg.step;
        debug_msg.data.resize(nbytes);
        std::memcpy(debug_msg.data.data(), continuous.ptr<float>(0), nbytes);
        publisher->publish(debug_msg);
    }

    void publish_debug_image(const std::vector<float>& data, int width, int height,
                             const sensor_msgs::msg::Image& source_msg,
                             const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
        if (!debug_vis_ || !publisher) {
            return;
        }
        sensor_msgs::msg::Image debug_msg;
        debug_msg.header = source_msg.header;
        debug_msg.height = static_cast<uint32_t>(height);
        debug_msg.width = static_cast<uint32_t>(width);
        debug_msg.encoding = "32FC1";
        debug_msg.is_bigendian = false;
        debug_msg.step = static_cast<uint32_t>(width * sizeof(float));
        debug_msg.data.resize(data.size() * sizeof(float));
        std::memcpy(debug_msg.data.data(), data.data(), data.size() * sizeof(float));
        publisher->publish(debug_msg);
    }

    // ---- History stack ----
    std::vector<float> build_depth_history(const std::vector<float>& latest_frame) {
        const size_t history_capacity = static_cast<size_t>(depth_history_taps_.front() + 1);
        if (depth_history_frames_.empty()) {
            depth_history_frames_.assign(history_capacity, latest_frame);
        } else {
            depth_history_frames_.push_back(latest_frame);
            while (depth_history_frames_.size() > history_capacity) {
                depth_history_frames_.pop_front();
            }
        }

        std::vector<float> history(
            static_cast<size_t>(depth_obs_num_) * depth_history_taps_.size(), 0.0f);
        const int64_t newest_idx = static_cast<int64_t>(depth_history_frames_.size()) - 1;
        for (size_t i = 0; i < depth_history_taps_.size(); ++i) {
            const int64_t history_idx = std::max<int64_t>(0, newest_idx - depth_history_taps_[i]);
            const auto& source = depth_history_frames_[static_cast<size_t>(history_idx)];
            std::copy_n(source.begin(), depth_obs_num_, history.begin() + i * depth_obs_num_);
        }
        return history;
    }

    // ---- Encode (or pass-through stacked frames) ----
    std::vector<float> encode_depth_history(const std::vector<float>& history) {
        if (!use_depth_encoder_) {
            return history;
        }
        std::copy(history.begin(), history.end(), encoder_ctx_.input_buffer.begin());
        encoder_ctx_.session->Run(Ort::RunOptions{nullptr}, encoder_ctx_.input_names_raw.data(),
                                  encoder_ctx_.input_tensor.get(), encoder_ctx_.num_inputs,
                                  encoder_ctx_.output_names_raw.data(), encoder_ctx_.output_tensor.get(),
                                  encoder_ctx_.num_outputs);
        return encoder_ctx_.output_buffer;
    }

   public:
    // ---- Runtime API ----
    // Orchestrate: cached image -> preprocess -> history -> encode.
    bool process_latest(std::vector<float>& output) {
        sensor_msgs::msg::Image::SharedPtr image;
        {
            std::unique_lock<std::mutex> lock(image_mutex_);
            image = latest_depth_image_;
        }
        if (!image) {
            return false;
        }

        std::unique_lock<std::mutex> process_lock(process_mutex_);
        std::vector<float> latest_frame = preprocess_depth_image(*image);
        if (latest_frame.empty()) {
            return false;
        }
        std::vector<float> history = build_depth_history(latest_frame);
        output = encode_depth_history(history);
        return true;
    }

    void reset() {
        std::unique_lock<std::mutex> lock(process_mutex_);
        depth_history_frames_.clear();
        std::fill(encoder_ctx_.input_buffer.begin(), encoder_ctx_.input_buffer.end(), 0.0f);
        std::fill(encoder_ctx_.output_buffer.begin(), encoder_ctx_.output_buffer.end(), 0.0f);
    }
   private:
    // Camera callback: cache latest frame only (no history/encode).
    void depth_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!msg) {
            return;
        }
        std::unique_lock<std::mutex> lock(image_mutex_);
        latest_depth_image_ = msg;
    }

    rclcpp::Node& node_;
    Ort::Env& env_;

    std::string depth_topic_;
    std::string model_dir_;
    std::string depth_encoder_model_name_;
    std::string depth_encoder_model_path_;
    std::string debug_depth_vis_topic_;
    bool use_depth_encoder_ = true;
    bool debug_vis_ = false;
    int depth_encoder_output_dim_ = 128;
    int depth_input_width_ = 640;
    int depth_input_height_ = 360;
    int depth_width_ = 64;
    int depth_height_ = 36;
    int depth_crop_up_ = 18;
    int depth_crop_down_ = 0;
    int depth_crop_left_ = 16;
    int depth_crop_right_ = 16;
    int depth_obs_width_ = 0;
    int depth_obs_height_ = 0;
    std::vector<int64_t> depth_history_taps_{35, 30, 25, 20, 15, 10, 5, 0};
    int depth_obs_num_ = 0;
    float depth_min_distance_ = 0.0f;
    float depth_max_distance_ = 2.5f;
    float depth_inpaint_distance_ = 0.2f;
    double depth_inpaint_radius_ = 3.0;
    int depth_gaussian_kernel_size_ = 3;
    double depth_gaussian_sigma_ = 1.0;

    Ort::AllocatorWithDefaultOptions allocator_;
    ModelContext encoder_ctx_;

    std::mutex image_mutex_;
    std::mutex process_mutex_;
    std::deque<std::vector<float>> depth_history_frames_;
    sensor_msgs::msg::Image::SharedPtr latest_depth_image_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_downsample_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_crop_publisher_;
};

DepthProvider::DepthProvider(rclcpp::Node& node, Ort::Env& env, std::size_t expected_output_size)
    : impl_(std::make_unique<Impl>(node, env, expected_output_size)) {}

DepthProvider::~DepthProvider() = default;

bool DepthProvider::process_latest(std::vector<float>& output) {
    return impl_->process_latest(output);
}

void DepthProvider::reset() {
    impl_->reset();
}
