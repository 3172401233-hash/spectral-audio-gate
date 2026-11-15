# spectral-audio-gate
Python 频谱噪声门库，处理 STFT、噪声底估计、频带门控和重叠相加

## 功能

- 单声道 float 输入，可配置窗长（`fft_size`）、hop（`hop_size`）与窗函数
- 对数间隔频带内独立估计噪声底并独立门控
- 软拐点（soft-knee）增益映射 + 每频带 attack/release 包络平滑
- 增益为实数非负值，只缩放幅度、保持原相位
- overlap-add 重建，带逐样本归一化（任意满足 COLA 的窗/hop 组合）
- 流式 `process_block()` / `flush()` 与一次性 `process()`，分块与整段结果逐样本一致

## 快速开始

```python
import numpy as np
from spectral_gate import SpectralGate

gate = SpectralGate(sample_rate=48000, fft_size=1024, hop_size=256)
y = gate.process(x)            # 一次性处理

# 流式分块处理（状态跨块保留：分析窗重叠、噪声底与增益包络）
gate = SpectralGate(sample_rate=48000)
for block in stream:
    out_block = gate.process_block(block)
tail = gate.flush()            # 处理残余输入并输出窗尾
```

## 窗函数

内置周期（periodic）窗：`hann`（默认）、`hamming`、`blackman`、`rect`，
也可传入长度为 `fft_size` 的自定义窗数组。分析与合成使用同一窗，
等效重叠权重为窗的平方。构造时会校验窗/hop 组合的 COLA 约束
（每个输出位置的重叠权重和必须大于零），不满足时抛出 `ValueError`。
注意 Hann 等窗在边界处权重为零，信号最前端的个别样本权重为零、输出为零，
属于正常现象。

## 延迟与输出长度

- 流式处理下，位置 `p` 的输出在覆盖它的所有分析窗处理完毕后才能发出，
  最坏情况延迟为 `fft_size - 1` 个样本（`SpectralGate.latency`）。
- 输出与输入逐样本对齐（位置 `k` 对应输入 `k`），`flush()` 之后
  输出尾部会超出输入末尾，总长为 `last_frame_start + fft_size`。

## 噪声底估计

- 每帧对每个频带计算带内 RMS 幅度（RMS 而非均值，避免宽频带稀释纯音峰值）。
- 噪声底为每频带的非对称指数跟踪：当前值低于噪声底时快速下降
  （`noise_fall_ms`，最小值统计风格），高于噪声底时缓慢上升
  （`noise_rise_ms`）。因此稳态噪声会被吸收为噪声底而被衰减；
  持续时间远超 `noise_rise_ms` 的声音也会逐渐被当作噪声——这是
  最小值跟踪类估计器的固有特性，可调大 `noise_rise_ms` 缓解。
- 首帧用于初始化噪声底，即默认假设信号开头（或低电平段）为噪声学习期。

## 增益计算

- 门限：`threshold_db` 为相对于噪声底的开启电平；`knee_db` 为软拐点宽度，
  增益在门限到门限 + `knee_db` 之间从 `min_gain_db` 平滑过渡到 0 dB。
- `min_gain_db`（默认 -24 dB）是最大衰减量而非完全清零，避免乐器尾音
  被硬切断产生金属感；软拐点与 attack/release 平滑避免底噪呼吸感。
- 每频带增益包络按帧做 attack（开）/ release（关）指数平滑，
  再映射到该频带所有 FFT bin 上，乘以复数频谱（实数增益，相位不变）。

## 参数校验

非法参数（非正采样率、`hop_size` 超出 `[1, fft_size]`、未知窗名、
违反 COLA 的窗/hop 组合、负的时间常数等）在构造时抛出 `ValueError`；
非一维或含非有限值的输入在 `process_block` 时抛出 `ValueError`。

## 测试

```
python -m unittest
```

覆盖：分块一致性、门限启停（稳态噪声衰减 / 突发人声重新打开 / 短脉冲通过）、
频带独立、相位保持、静音与空输入、错误参数。
