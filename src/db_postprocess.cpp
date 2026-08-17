#include "db_postprocess.h" 

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>  // OpenCV 5: minAreaRect / contourArea live here
#include <opencv2/core/utility.hpp>

DBPostProcess::DBPostProcess(float thresh, float box_thresh, float unclip_ratio,
                             int max_candidates)
    : thresh_(thresh), box_thresh_(box_thresh), unclip_ratio_(unclip_ratio),
      max_candidates_(max_candidates) {}

float DBPostProcess::box_score_fast(const cv::Mat& bitmap,
                                    const std::vector<cv::Point>& contour) {
    cv::Rect rect = cv::boundingRect(contour);
    rect &= cv::Rect(0, 0, bitmap.cols, bitmap.rows);
    if (rect.empty()) return 0.0f;

    cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8UC1);
    std::vector<cv::Point> shifted;
    shifted.reserve(contour.size());
    for (const auto& p : contour)
        shifted.emplace_back(p.x - rect.x, p.y - rect.y);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted}, 1);
    return static_cast<float>(cv::mean(bitmap(rect), mask)[0]);
}

std::vector<cv::Point2f> DBPostProcess::order_clockwise(const std::vector<cv::Point2f>& pts) {
    cv::Point2f c(0, 0);
    for (auto p : pts) c += p;
    c *= (1.0f / static_cast<float>(pts.size()));

    std::vector<cv::Point2f> out = pts;
    std::sort(out.begin(), out.end(), [&](const auto& a, const auto& b) {
        return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
    });
    // Force clockwise in image coordinates.
    double area = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const auto& p = out[i];
        const auto& q = out[(i + 1) % out.size()];
        area += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    if (area < 0.0) std::reverse(out.begin(), out.end());
    return out;
}

std::vector<cv::Point2f> DBPostProcess::offset_polygon(const std::vector<cv::Point2f>& poly,
                                                       float distance) {
    if (poly.size() != 4) return poly;

    // Polygon is clockwise in image coordinates. For each directed edge, the
    // right-hand normal is approximately outward in this coordinate system.
    struct Line { cv::Point2f p; cv::Point2f d; } lines[4];
    for (int i = 0; i < 4; ++i) {
        cv::Point2f p = poly[i];
        cv::Point2f q = poly[(i + 1) % 4];
        cv::Point2f d = q - p;
        float len = std::sqrt(d.dot(d));
        if (len < 1e-4f) return poly;
        cv::Point2f n(d.y / len, -d.x / len);
        lines[i] = {p + n * distance, d};
    }

    auto intersection = [](const Line& a, const Line& b, cv::Point2f& out) -> bool {
        float cross = a.d.x * b.d.y - a.d.y * b.d.x;
        if (std::abs(cross) < 1e-6f) return false;
        cv::Point2f r = b.p - a.p;
        float t = (r.x * b.d.y - r.y * b.d.x) / cross;
        out = a.p + a.d * t;
        return true;
    };

    std::vector<cv::Point2f> out(4);
    for (int i = 0; i < 4; ++i) {
        if (!intersection(lines[(i + 3) % 4], lines[i], out[i])) return poly;
    }
    return out;
}

std::vector<DetBox> DBPostProcess::operator()(const cv::Mat& prob,
                                               float scale_x, float scale_y,
                                               const cv::Size& original_size) const {
    CV_Assert(prob.type() == CV_32FC1);
    cv::Mat bitmap;
    cv::threshold(prob, bitmap, thresh_, 1.0, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8UC1, 255.0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const int limit = std::min<int>(static_cast<int>(contours.size()), max_candidates_);

    // Per-contour work (bounding-rect masking, minAreaRect, unclip offset) is
    // independent across contours, so it parallelizes cleanly across the
    // available cores. Each slot is written by exactly one iteration, so no
    // locking is needed; we compact the "valid" entries afterward.
    std::vector<DetBox> slots(limit);
    std::vector<uint8_t> valid(limit, 0);

    cv::parallel_for_(cv::Range(0, limit), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            if (contours[i].size() < 4) continue;
            float score = box_score_fast(prob, contours[i]);
            if (score < box_thresh_) continue;

            cv::RotatedRect rr = cv::minAreaRect(contours[i]);
            cv::Point2f pts4[4];
            rr.points(pts4);
            std::vector<cv::Point2f> poly(pts4, pts4 + 4);
            poly = order_clockwise(poly);

            double area = std::abs(cv::contourArea(poly));
            double perimeter = cv::arcLength(poly, true);
            if (perimeter < 1e-5) continue;
            float distance = static_cast<float>(area * unclip_ratio_ / perimeter);
            poly = offset_polygon(poly, distance);

            for (auto& p : poly) {
                p.x /= scale_x;
                p.y /= scale_y;
                p.x = std::clamp(p.x, 0.0f, static_cast<float>(original_size.width - 1));
                p.y = std::clamp(p.y, 0.0f, static_cast<float>(original_size.height - 1));
            }

            double area2 = std::abs(cv::contourArea(poly));
            if (area2 < 4.0) continue;

            slots[i] = DetBox{poly, score};
            valid[i] = 1;
        }
    });

    std::vector<DetBox> results;
    results.reserve(limit);
    for (int i = 0; i < limit; ++i) {
        if (valid[i]) results.push_back(std::move(slots[i]));
    }

    std::sort(results.begin(), results.end(), [](const DetBox& a, const DetBox& b) {
        float ay = 0, by = 0, ax = 0, bx = 0;
        for (auto p : a.points) { ax += p.x; ay += p.y; }
        for (auto p : b.points) { bx += p.x; by += p.y; }
        ax /= 4; ay /= 4; bx /= 4; by /= 4;
        if (std::abs(ay - by) > 10.0f) return ay < by;
        return ax < bx;
    });
    return results;
}
