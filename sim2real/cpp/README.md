# C++ 推理（占位，暂未实现）

本目录将放置 sim2real 的 **C++ 版策略推理代码**，与 [`../python/`](../python/) 的 Python 版并列。

两版共用同一份配置 [`../config/sim2real.yaml`](../config/sim2real.yaml)，必须保证：

- 串口/帧协议参数（port、baudrate、SOF/EOF、`use_crc8`、CRC 多项式、`uplink_format`/`downlink_format`）；
- 观测拼接顺序与 `observation_scales`、`clipping`；
- 网络结构（`num_obs`、`latent_dim`、各层宽度、`activation`、`model_path`）；

与 Python 版逐项一致，否则两套部署行为会不同。

Python 版可作为 C++ 实现的参考对照：
[`serial_comm.py`](../python/serial_comm.py)（串口与帧协议）、
[`controller.py`](../python/controller.py)（观测拼装/缩放/历史）、
[`policy.py`](../python/policy.py)（encoder+actor 前向）。
