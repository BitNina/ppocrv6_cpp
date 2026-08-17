cmake_minimum_required(VERSION 3.20)

project(ppocrv6_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ONNXRUNTIME_ROOT "C:/Lib/onnxruntime-win-x64-1.29.0")
set(OpenCV_DIR "C:/Lib/opencv/build")

if(MSVC)
    add_compile_options(/utf-8 /W4 /EHsc)
endif()

# Expected environment variables:
#   ONNXRUNTIME_ROOT = path containing include/ and lib/
#   OpenCV_DIR       = directory containing OpenCVConfig.cmake

if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory.")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find ONNX Runtime headers/library under ${ONNXRUNTIME_ROOT}")
endif()

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    geometry
)

add_executable(ppocrv6
    src/main.cpp
    src/ppocr.cpp
    src/db_postprocess.cpp
)

target_include_directories(ppocrv6 PRIVATE
    include
    "${ONNXRUNTIME_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ppocrv6 PRIVATE
    "${ONNXRUNTIME_LIBRARY}"
    ${OpenCV_LIBS}
)
# ------------------------------------------------------------
# Runtime DLLs
# ------------------------------------------------------------

if(WIN32)

    # ONNX Runtime
    add_custom_command(TARGET ppocrv6 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:ppocrv6>/onnxruntime.dll"
    )

    # OpenCV
    file(GLOB OPENCV_DLLS
        "${OpenCV_INSTALL_PATH}/x64/vc16/bin/*.dll"
    )

    foreach(OPENCV_DLL ${OPENCV_DLLS})
        add_custom_command(TARGET ppocrv6 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL}"
                "$<TARGET_FILE_DIR:ppocrv6>"
        )
    endforeach()

endif()