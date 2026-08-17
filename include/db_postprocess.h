#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct DetBox {
    std::vector<cv::Point2f> points; // 4 points, clockwise
    float score = 0.0f;
};

class DBPostProcess {
public:
    DBPostProcess(float thresh = 0.20f,
                  float box_thresh = 0.45f,
                  float unclip_ratio = 1.40f,
                  int max_candidates = 3000);

    std::vector<DetBox> operator()(const cv::Mat& prob,
                                   float scale_x,
                                   float scale_y,
                                   const cv::Size& original_size) const;

private:
    float thresh_;
    float box_thresh_;
    float unclip_ratio_;
    int max_candidates_;

    static float box_score_fast(const cv::Mat& bitmap,
                                const std::vector<cv::Point>& contour);
    static std::vector<cv::Point2f> order_clockwise(const std::vector<cv::Point2f>& pts);
    static std::vector<cv::Point2f> offset_polygon(const std::vector<cv::Point2f>& poly,
                                                   float distance);
};
