#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "db_postprocess.h"

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
