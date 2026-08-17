#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "db_postprocess.h"

// Minimal fixed-size thread pool used to run recognition inference for
// several detected text boxes concurrently. ONNX Runtime CPU sessions
// support concurrent Run() calls from multiple threads, so as long as each
// session is configured with a single intra-op thread (see ppocr.cpp), this
// keeps total inference parallelism matched to the physical core count
// instead of oversubscribing it.
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Submits a batch of jobs and blocks until all of them complete.
    void run_all(const std::vector<std::function<void()>>& jobs);

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_tasks_;
    std::condition_variable cv_done_;
    size_t pending_ = 0;
    bool stop_ = false;

    void worker_loop();
};

struct OCRResult {
    std::string text;
    float score = 0.0f;
    std::vector<cv::Point2f> box;
};

class PP_OCRv6 {
public:
    PP_OCRv6(const std::string& det_model,
             const std::string& rec_model,
             const std::string& dict_file,
             int threads = 4);

    std::vector<OCRResult> run(const cv::Mat& image);

private:
    Ort::Env env_;
    Ort::SessionOptions det_options_;
    Ort::SessionOptions rec_options_;
    Ort::Session det_session_{nullptr};
    Ort::Session rec_session_{nullptr};

    std::string det_input_name_;
    std::string rec_input_name_;
    std::string det_output_name_;
    std::string rec_ctc_output_name_;

    std::vector<std::string> dict_;
    DBPostProcess db_;
    ThreadPool rec_pool_;

    static std::vector<std::string> get_node_names(Ort::Session& session, bool input);
    static cv::Mat resize_det(const cv::Mat& image, float& scale_x, float& scale_y);
    static std::vector<float> normalize_chw(const cv::Mat& bgr);

    std::vector<DetBox> detect(const cv::Mat& image);
    std::pair<std::string, float> recognize(const cv::Mat& crop);

    static std::vector<float> tensor_to_vector(const Ort::Value& value);
    static std::vector<int64_t> tensor_shape(const Ort::Value& value);
    static cv::Mat four_point_crop(const cv::Mat& image, const std::vector<cv::Point2f>& box);
    static std::vector<float> sigmoid_if_needed(std::vector<float> v);
};
