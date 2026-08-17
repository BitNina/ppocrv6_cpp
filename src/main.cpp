#include "ppocr.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <string>

#include <chrono>

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    try {
        std::string image_path = argv[1];
        std::string det_model = argc > 2 ? argv[2] : "models/det.onnx";
        std::string rec_model = argc > 3 ? argv[3] : "models/rec.onnx";
        std::string dict_file = argc > 4 ? argv[4] : "models/ppocrv6_dict.txt";

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Cannot read image: " << image_path << '\n';
            return 2;
        }

        PP_OCRv6 ocr(det_model, rec_model, dict_file, 4);
        
        auto start = std::chrono::steady_clock::now();

        auto results = ocr.run(image);

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "OCR time: "<< elapsed_ms<< " ms"<< std::endl;

        cv::Mat vis = image.clone();
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            for (int j = 0; j < 4; ++j)
                cv::line(vis, r.box[j], r.box[(j+1)%4], cv::Scalar(0,255,0), 2);

            std::cout << '[' << i << "] " << r.text
                      << "  score=" << r.score << '\n';
        }
        cv::imwrite("ocr_result.jpg", vis);
        std::cout << "Saved visualization to ocr_result.jpg\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 4;
    }
    return 0;
}
