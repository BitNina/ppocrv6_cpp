#include "ppocr.h" 

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>  // OpenCV 5 module for computational geometry

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// ONNX Runtime's Session constructor takes ORTCHAR_T*, which is wchar_t on
// Windows and char everywhere else. This converts a UTF-8 std::string into
// whatever that platform expects, so the same source compiles and works on
// both Windows and Linux.
#ifdef _WIN32
std::wstring ToOrtPath(const std::string& s) {
    if (s.empty()) return {};

    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) throw std::runtime_error("MultiByteToWideChar failed for model path.");

    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}
#else
std::string ToOrtPath(const std::string& s) {
    return s;  // Linux/macOS: ORTCHAR_T == char, and the filesystem is UTF-8 already.
}
#endif

}  // namespace

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) num_threads = 1;
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back(&ThreadPool::worker_loop, this);
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_tasks_.notify_all();
    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_tasks_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            job = std::move(tasks_.front());
            tasks_.pop();
        }
        job();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --pending_;
            if (pending_ == 0) cv_done_.notify_all();
        }
    }
}

void ThreadPool::run_all(const std::vector<std::function<void()>>& jobs) {
    if (jobs.empty()) return;
    if (workers_.empty()) {
        for (auto& j : jobs) j();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = jobs.size();
        for (auto& j : jobs) tasks_.push(j);
    }
    cv_tasks_.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this] { return pending_ == 0; });
    }
}

PP_OCRv6::PP_OCRv6(const std::string& det_model,
                   const std::string& rec_model,
                   const std::string& dict_file,
                   int threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PP-OCRv6"),
      db_(0.20f, 0.45f, 1.40f, 3000),
      rec_pool_(static_cast<size_t>(std::max(1, threads))) {
    if (threads < 1) threads = 1;

    det_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    rec_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Both sessions run a single graph per call, so there's no benefit to
    // the parallel executor; sequential mode has lower overhead.
    det_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    rec_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    det_options_.SetInterOpNumThreads(1);
    rec_options_.SetInterOpNumThreads(1);

    // Detection runs once per image, so it gets to use all cores for
    // intra-op parallelism within that single call.
    det_options_.SetIntraOpNumThreads(threads);
    // Recognition instead runs once per detected text box, often many times
    // per image. Instead of parallelizing *inside* each small inference
    // (high overhead relative to the tiny 48xW crops), we run `threads`
    // single-threaded recognition calls concurrently via rec_pool_ below.
    // This keeps total CPU usage matched to the physical core count without
    // oversubscribing it.
    rec_options_.SetIntraOpNumThreads(1);

    det_session_ = Ort::Session(env_, ToOrtPath(det_model).c_str(), det_options_);
    rec_session_ = Ort::Session(env_, ToOrtPath(rec_model).c_str(), rec_options_);

    auto det_inputs = get_node_names(det_session_, true);
    auto rec_inputs = get_node_names(rec_session_, true);
    auto det_outputs = get_node_names(det_session_, false);
    auto rec_outputs = get_node_names(rec_session_, false);
    if (det_inputs.empty() || rec_inputs.empty() || det_outputs.empty() || rec_outputs.empty())
        throw std::runtime_error("Invalid ONNX model: missing inputs/outputs.");

    det_input_name_ = det_inputs.front();
    rec_input_name_ = rec_inputs.front();
    det_output_name_ = det_outputs.front();

    // PP-OCRv6 recognition is trained with a CTC head. The ONNX export can
    // expose more than one output; select the rank-3 floating output with the
    // largest class dimension as the CTC logits/probabilities.
    size_t best = 0;
    size_t best_score = 0;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < rec_outputs.size(); ++i) {
        Ort::AllocatedStringPtr name = rec_session_.GetOutputNameAllocated(i, allocator);
        auto ti = rec_session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = ti.GetShape();
        size_t score = 0;
        if (shape.size() == 3) {
            for (auto d : shape) if (d > 0) score = std::max(score, static_cast<size_t>(d));
        }
        if (score > best_score) { best_score = score; best = i; }
    }
    rec_ctc_output_name_ = rec_outputs[best];

    std::ifstream fin(dict_file, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open dictionary: " + dict_file);
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) dict_.push_back(line);
    }
    // Official PP-OCRv6 small/medium config uses use_space_char=True.
    dict_.push_back(" ");
}

std::vector<std::string> PP_OCRv6::get_node_names(Ort::Session& session, bool input) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = input ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Ort::AllocatedStringPtr name = input
            ? session.GetInputNameAllocated(i, allocator)
            : session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(name.get());
    }
    return names;
}

cv::Mat PP_OCRv6::resize_det(
    const cv::Mat& image,
    float& scale_x,
    float& scale_y
) {
    constexpr int limit = 960;
    constexpr int min_side = 32;

    const int src_w = image.cols;
    const int src_h = image.rows;

    float scale = 1.0f;

    int max_side = std::max(src_w, src_h);

    if (max_side > limit) {
        scale = static_cast<float>(limit) /
            static_cast<float>(max_side);
    }

    int new_w = std::max(
        min_side,
        static_cast<int>(std::round(src_w * scale))
    );

    int new_h = std::max(
        min_side,
        static_cast<int>(std::round(src_h * scale))
    );

    // Keep aspect ratio first.
    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(new_w, new_h),
        0,
        0,
        cv::INTER_LINEAR
    );

    // Pad to multiples of 32.
    int pad_w = (32 - (new_w % 32)) % 32;
    int pad_h = (32 - (new_h % 32)) % 32;

    if (pad_w != 0 || pad_h != 0) {
        cv::copyMakeBorder(
            resized,
            resized,
            0,
            pad_h,
            0,
            pad_w,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );
    }

    // Important:
    // Coordinates must be restored using the RESIZE scale,
    // not the padded dimensions.
    scale_x =
        static_cast<float>(new_w) /
        static_cast<float>(src_w);

    scale_y =
        static_cast<float>(new_h) /
        static_cast<float>(src_h);

    return resized;
}

std::vector<float> PP_OCRv6::normalize_chw(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stdv{0.229f, 0.224f, 0.225f};
    std::vector<float> out(static_cast<size_t>(rgb.rows) * rgb.cols * 3);
    size_t base = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < rgb.rows; ++y) {
            const auto* row = rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < rgb.cols; ++x) {
                out[base++] = (row[x][c] - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

std::vector<float> PP_OCRv6::tensor_to_vector(const Ort::Value& value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    size_t n = info.GetElementCount();
    const float* p = value.GetTensorData<float>();
    return std::vector<float>(p, p + n);
}

std::vector<int64_t> PP_OCRv6::tensor_shape(const Ort::Value& value) {
    return value.GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<float> PP_OCRv6::sigmoid_if_needed(std::vector<float> v) {
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    if (*mn >= -1e-5f && *mx <= 1.00001f) return v;
    for (float& x : v) x = 1.0f / (1.0f + std::exp(-x));
    return v;
}

std::vector<DetBox> PP_OCRv6::detect(const cv::Mat& image) {
    float sx = 1.0f, sy = 1.0f;
    cv::Mat resized = resize_det(image, sx, sy);
    std::vector<float> input = normalize_chw(resized);

    int64_t shape[4] = { 1, 3, resized.rows, resized.cols };
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 4);

    const char* in_names[] = { det_input_name_.c_str() };
    const char* out_names[] = { det_output_name_.c_str() };
    auto outputs = det_session_.Run(Ort::RunOptions{ nullptr }, in_names, &tensor, 1, out_names, 1);
    auto shape_out = tensor_shape(outputs.front());
    auto data = sigmoid_if_needed(tensor_to_vector(outputs.front()));

    int H = 0, W = 0;
    if (shape_out.size() == 4) {
        H = static_cast<int>(shape_out[2]);
        W = static_cast<int>(shape_out[3]);
    }
    else if (shape_out.size() == 3) {
        H = static_cast<int>(shape_out[1]);
        W = static_cast<int>(shape_out[2]);
    }
    else {
        throw std::runtime_error("Unexpected detection output rank.");
    }
    if (static_cast<size_t>(H) * W != data.size()) {
        throw std::runtime_error("Unexpected detection output dimensions.");
    }

    cv::Mat prob(H, W, CV_32FC1, data.data());
    auto boxes = db_(prob, static_cast<float>(W) / resized.cols,
        static_cast<float>(H) / resized.rows, image.size());

    // ===== 关键修复：把坐标从 resized 空间还原到原图空间 =====
    for (auto& box : boxes) {
        for (auto& p : box.points) {
            p.x /= sx;
            p.y /= sy;
        }
    }
    // =========================================================

    return boxes;
}

cv::Mat PP_OCRv6::four_point_crop(const cv::Mat& image, const std::vector<cv::Point2f>& box) {
    CV_Assert(box.size() == 4);
    std::vector<cv::Point2f> pts = box;   // db_postprocess 已保证顺时针，直接复用

    auto dist = [](cv::Point2f a, cv::Point2f b) { return cv::norm(a - b); };
    float w1 = dist(pts[0], pts[1]);
    float w2 = dist(pts[2], pts[3]);
    float h1 = dist(pts[0], pts[3]);
    float h2 = dist(pts[1], pts[2]);
    int w = std::max(1, static_cast<int>(std::round(std::max(w1, w2))));
    int h = std::max(1, static_cast<int>(std::round(std::max(h1, h2))));
    std::vector<cv::Point2f> dst{ {0,0}, {static_cast<float>(w - 1),0},
                                 {static_cast<float>(w - 1), static_cast<float>(h - 1)},
                                 {0, static_cast<float>(h - 1)} };

    cv::Mat M = cv::getPerspectiveTransform(pts, dst);
    cv::Mat crop;
    cv::warpPerspective(image, crop, M, cv::Size(w, h), cv::INTER_CUBIC,
        cv::BORDER_REPLICATE);
    return crop;
}

std::pair<std::string, float> PP_OCRv6::recognize(const cv::Mat& crop) {
    constexpr int H = 48;
    constexpr int W = 320;
    if (crop.empty()) return {"", 0.0f};

    cv::Mat img = crop;
    if (img.rows * 1.0 / img.cols > 1.5) {
        // PaddleOCR rotates a tall crop 90 degrees before recognition.
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    }
    float ratio = static_cast<float>(img.cols) / std::max(1, img.rows);
    int target_w = std::min(W, std::max(1, static_cast<int>(std::ceil(H * ratio))));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, H), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded = cv::Mat::zeros(H, W, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, target_w, H)));

    // Recognition preprocessing differs from detection in PaddleOCR: scale to
    // [-1, 1] per channel after RGB conversion.
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0);
    std::vector<float> input(static_cast<size_t>(3) * H * W);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                input[idx++] = rgb.at<cv::Vec3f>(y,x)[c];

    int64_t shape[4] = {1, 3, H, W};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 4);
    const char* in_names[] = {rec_input_name_.c_str()};
    const char* out_names[] = {rec_ctc_output_name_.c_str()};
    auto outputs = rec_session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    auto shape_out = tensor_shape(outputs.front());
    auto logits = tensor_to_vector(outputs.front());
    if (shape_out.size() != 3 || shape_out[0] != 1)
        throw std::runtime_error("Unexpected recognition output rank/shape.");

    int64_t T = shape_out[1];
    int64_t C = shape_out[2];
    if (static_cast<size_t>(T*C) != logits.size())
        throw std::runtime_error("Unexpected recognition output element count.");

    // CTC decode: blank is index 0; collapse repeated indices.
    std::string text;
    float sum_prob = 0.0f;
    int count = 0;
    int prev = -1;
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits.data() + t*C;
        int argmax = 0;
        float best = row[0];
        for (int64_t c = 1; c < C; ++c) {
            if (row[c] > best) { best = row[c]; argmax = static_cast<int>(c); }
        }
        // If model export has probabilities use them directly; otherwise softmax
        // the row to obtain a confidence score.
        float p;
        bool row_prob = true;
        if (best < -1e-5f || best > 1.00001f) row_prob = false;
        if (row_prob) {
            p = std::clamp(best, 0.0f, 1.0f);
        } else {
            float mx = row[0];
            for (int64_t c = 1; c < C; ++c) mx = std::max(mx, row[c]);
            float denom = 0.0f;
            for (int64_t c = 0; c < C; ++c) denom += std::exp(row[c]-mx);
            p = std::exp(best-mx) / denom;
        }
        if (argmax != 0 && argmax != prev) {
            int dict_index = argmax - 1;
            if (dict_index >= 0 && dict_index < static_cast<int>(dict_.size()))
                text += dict_[dict_index];
            sum_prob += p;
            ++count;
        }
        prev = argmax;
    }
    return {text, count ? sum_prob / count : 0.0f};
}

std::vector<OCRResult> PP_OCRv6::run(const cv::Mat& image) {
    if (image.empty()) throw std::runtime_error("Input image is empty.");
    auto boxes = detect(image);
    if (boxes.empty()) return {};

    // Cropping is cheap (warpPerspective on small regions); do it up front
    // so the thread pool jobs below are pure inference work.
    std::vector<cv::Mat> crops(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
        crops[i] = four_point_crop(image, boxes[i].points);

    // Recognition is independent per box (rec_session_ is configured with a
    // single intra-op thread, see the constructor), so fan it out across
    // the pool: this is where multi-core speedup actually shows up on
    // images with several text lines.
    std::vector<std::string> texts(boxes.size());
    std::vector<float> scores(boxes.size(), 0.0f);

    std::vector<std::function<void()>> jobs;
    jobs.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        jobs.emplace_back([this, &crops, &texts, &scores, i] {
            auto [text, rec_score] = recognize(crops[i]);
            texts[i] = std::move(text);
            scores[i] = rec_score;
        });
    }
    rec_pool_.run_all(jobs);

    std::vector<OCRResult> results;
    results.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!texts[i].empty())
            results.push_back({std::move(texts[i]), scores[i], boxes[i].points});
    }
    return results;
}
