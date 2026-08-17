# PP-OCRv6 + ONNX Runtime + C++17 + OpenCV (Windows)

这是一个不依赖 Python / PaddlePaddle 的 PP-OCRv6 small C++ 推理工程：

```text
image
  -> OpenCV decode
  -> PP-OCRv6 small det ONNX
  -> DB postprocess
  -> perspective crop
  -> PP-OCRv6 small rec ONNX
  -> CTC decode
  -> text + score + 4-point box
```

## 1. 模型

工程默认使用：

- `PP-OCRv6_small_det_onnx/inference.onnx`
- `PP-OCRv6_small_rec_onnx/inference.onnx`
- `ppocrv6_dict.txt`

下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\download_models.ps1
```

注意：ONNX 模型不随源码压缩包重复打包，脚本会从官方仓库下载。

## 2. Windows 依赖

需要：

- Visual Studio 2022，Desktop development with C++
- CMake >= 3.20
- OpenCV 4.x
- ONNX Runtime 1.x CPU package

目录可以类似：

```text
D:\libs\onnxruntime-win-x64-1.xx.x\
  include\onnxruntime_cxx_api.h
  lib\onnxruntime.lib
  lib\onnxruntime.dll

D:\libs\opencv\build\
  OpenCVConfig.cmake
```

设置环境变量：

```powershell
$env:ONNXRUNTIME_ROOT="D:\libs\onnxruntime-win-x64-1.xx.x"
```

然后让 CMake 找到 OpenCV，例如：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR="D:\libs\opencv\build"
```

构建：

```powershell
cmake --build build --config Release
```

运行：

```powershell
.\build\Release\ppocrv6.exe .\test.jpg
```

或者显式指定路径：

```powershell
.\build\Release\ppocrv6.exe `
  .\test.jpg `
  .\models\det.onnx `
  .\models\rec.onnx `
  .\models\ppocrv6_dict.txt
```

程序会：

1. 在控制台输出文字和识别置信度。
2. 输出检测框。
3. 生成 `ocr_result.jpg`。

## 3. 预处理 / 后处理

Detection：

- BGR -> RGB
- `/255`
- mean `[0.485, 0.456, 0.406]`
- std `[0.229, 0.224, 0.225]`
- H/W 对齐到 32 的倍数
- 默认最长边 960
- DB `thresh=0.20`
- DB `box_thresh=0.45`
- DB `unclip_ratio=1.40`

Recognition：

- 透视裁剪
- resize 到高度 48
- 最大宽度 320
- 右侧 zero padding
- RGB
- `(x / 127.5) - 1`
- CTC greedy decode
- blank index = 0
- `use_space_char = true`

## 4. 关于 DB unclip

`db_postprocess.cpp` 没有依赖 Paddle/PyClipper，而是对四边形做几何 offset，所以它是一个轻量 C++ 实现。

这意味着它与 PaddleOCR Python 中 Vatti/PyClipper 的 polygon offset 并非逐像素完全一致，但一般文本框情况下可直接使用。

如果你要做到与 PaddleOCR 官方输出尽可能 bit-level/box-level 接近，可以把这里替换成 Clipper2 的 polygon offset 实现。

## 5. CUDA

默认是 CPU。

如果你安装的是 ONNX Runtime GPU 包，可以：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR="D:\libs\opencv\build" `
  -DUSE_CUDA=ON
```

然后确保 ONNX Runtime GPU 对应的 CUDA/cuDNN DLL 能被 Windows 找到。

## 6. 代码结构

```text
ppocrv6_cpp/
├─ CMakeLists.txt
├─ README.md
├─ download_models.ps1
├─ include/
│  ├─ db_postprocess.h
│  └─ ppocr.h
├─ src/
│  ├─ db_postprocess.cpp
│  ├─ main.cpp
│  └─ ppocr.cpp
└─ models/
   ├─ det.onnx
   ├─ rec.onnx
   ├─ ppocrv6_dict.txt
   ├─ det_inference.yml
   └─ rec_inference.yml
```
