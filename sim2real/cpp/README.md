# sim2real C++ 部署（ONNX Runtime + POSIX 串口）

Python 版（[`../python/`](../python/)）的 C++ 对应实现，与之共用配置
[`../config/sim2real.yaml`](../config/sim2real.yaml)。在仅有 CPU 的 NUC 上轻量运行：
POSIX termios 串口 + ONNX Runtime CPU 推理，不依赖 ROS2。

## 模块（对应 Python）

| C++ | Python | 职责 |
|---|---|---|
| `config.{hpp,cpp}` | `config.py` | 加载 sim2real.yaml、解析帧格式与相对仓库根的模型路径、校验维度 |
| `protocol.{hpp,cpp}` | serial_comm 的 decode/encode | 21 维上行解码、6 维下行编码（小端，按位读写，不裸 memcpy 结构体） |
| `frame_codec.{hpp,cpp}` | `FrameCodec` | `[SOF][payload][CRC8(可选)][EOF]`、找帧/重同步/断帧重组 |
| `serial_transport.{hpp,cpp}` | `SerialTransport` | termios 8N1、raw、`VMIN=0/VTIME=0` 非阻塞读 |
| `robot_serial_link.{hpp,cpp}` | `RobotSerialLink` | `poll()` 返回最新帧、`sendAction()` 下发 |
| `controller.{hpp,cpp}` | `Sim2RealController` | 21→27 维观测缩放/裁剪、5 帧历史、last_action |
| `onnx_policy_runner.{hpp,cpp}` | `policy.py` | ONNX Runtime CPU 推理（无 ORT 时编译为报错桩） |
| `deploy.cpp` | `deploy.py` | 固定频率主循环、丢帧重发、链路丢失停机、退出下发零动作 |

`transport.hpp` 提供 `ITransport` 抽象，使 `RobotSerialLink` 可用 mock 在无硬件时测试。

## 依赖

- C++17 编译器、CMake ≥ 3.16
- **yaml-cpp**（缺失：`sudo apt install libyaml-cpp-dev`）
- **ONNX Runtime C++**（可选）：缺失时 policy 编译为桩、构建仍成功，但运行推理会明确报错。
  安装：从 https://github.com/microsoft/onnxruntime/releases 下载 Linux 发行包，
  解压到 `/usr/local` 或重新配置时传 `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`。

## 构建与测试

```bash
cd sim2real/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`compile_commands.json` 会生成在 `build/`（供 VSCode/clangd 用，可软链到 `cpp/`）。

## 导出 ONNX 模型

C++ policy 需要 ONNX，而非 PyTorch `.pt`。先导出一次：

```bash
conda run -n isaac_gym python sim2real/python/export_onnx.py
```

它把 `network.model_path` 指向的检查点导出为单图 ONNX（输入 `obs`[1,27] 与
`obs_history`[1,135]，输出 `action`[1,6]），写到 `network.onnx_model_path`
（默认 `sim2real/model/policy.onnx`）。导出前 C++ policy 会明确报“模型缺失/需导出”。

## 运行

```bash
./sim2real/cpp/build/deploy --config sim2real/config/sim2real.yaml
```

需 ONNX Runtime 构建 + 已导出 `.onnx` + 串口可用。Ctrl-C 或链路丢失时下发零动作并关闭串口。
