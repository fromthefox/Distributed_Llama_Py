# Distributed-Llama-Py

## 模块概述 / Module Overview

本模块是LLM自动部署项目的核心分布式推理组件，负责实现Llama-3模型的不均匀分布式推理。提供API接口，支持动态任务重分配功能。

This module is the core distributed inference component of the LLM auto-deployment project, responsible for implementing uneven distributed inference of Llama-3 models. It provides API interfaces and supports dynamic task reallocation functionality.

## 核心功能 / Core Features

### 1. 分布式模型推理 / Distributed Model Inference
- 支持Llama-3模型的分片推理
- 不均匀任务分配策略
- 服务器-工作节点架构

### 2. 网络通信 / Network Communication
- TCP Socket通信机制
- 高效数据传输协议
- 多线程并发处理

### 3. 动态重分配支持 / Dynamic Reallocation Support
- **NEW**: Token级实时性能监控
- **NEW**: 支持推理过程中的任务重分配
- **NEW**: 与上级模块的性能追踪集成

## 文件结构 / File Structure

```
Distributed_Llama_Py/
├── model_inference_main_for_server.py  # 服务器端主程序
├── model_inference_main_for_worker.py  # 工作节点主程序
├── model_inference_framework.py       # 推理框架核心 (含动态分配)
├── model_inference_module.py          # 推理模块
├── socket_server.py                   # Socket服务器
├── init.py                           # 初始化模块
├── user_config.ini                   # 用户配置文件
└── ...
```

## API接口 / API Interface

### 服务器端接口 / Server Interface
```python
infenerce_main_for_server(
    allocation_list,           # 任务分配列表
    model_path,               # 模型路径
    tokenizer_path,           # 分词器路径
    config_path,              # 配置文件路径
    user_config_path,         # 用户配置路径
    dynamic_part,             # 动态权重部分
    nodes_info_dict,          # 节点信息字典
    performance_tracker=None, # 性能追踪器 (NEW)
    enable_dynamic_reallocation=False  # 启用动态重分配 (NEW)
)
```

### 工作节点接口 / Worker Interface
```python
inference_main_for_worker()
```

## 推理流程 / Inference Pipeline

1. **初始化阶段 / Initialization**
   - 加载模型、分词器、配置
   - 启动TCP服务器
   - 建立节点通信

2. **推理循环 / Inference Loop**
   - Token逐步生成
   - **NEW**: 性能数据收集
   - **NEW**: 动态权重更新
   - **NEW**: 任务重分配判断

3. **结果返回 / Result Return**
   - 完整文本输出
   - 性能统计信息

## 动态分配新特性 / Dynamic Allocation Features

### 1. 性能监控 / Performance Monitoring
- 计算时间统计：每个节点的推理耗时
- 通信时间统计：数据传输延迟
- 历史性能数据维护

### 2. 自适应调整 / Adaptive Adjustment
- 基于实际性能更新权重
- 阈值控制的重分配触发
- 最小Token数限制

### 3. 配置参数 / Configuration Parameters
```python
reallocation_threshold = 0.15    # 重分配阈值
min_tokens_before_reallocation = 1  # 最小Token数
performance_weight = 0.3         # 性能权重影响因子
```

## 使用示例 / Usage Example

```python
# 启用动态重分配的服务器端推理
result = infenerce_main_for_server(
    allocation_list=[47, 47, 17, 17],
    model_path="path/to/model.pth",
    tokenizer_path="path/to/tokenizer.model",
    config_path="path/to/config.json",
    user_config_path="path/to/user_config.ini",
    dynamic_part=np.array([0.5, 0.4, 0.1]),
    nodes_info_dict={
        'arithmetic': [603, 603, 2301, 2301],
        'memory': [8, 8, 350, 350],
        'bandwidth': [75, 75, 64, 64]
    },
    performance_tracker=PerformanceTracker(4),
    enable_dynamic_reallocation=True
)
```

## 配置文件说明 / Configuration File

### user_config.ini
```ini
[user_config]
input_text = Your input text here
max_token_length = 100
```

## 技术特点 / Technical Features

- **高效通信**: 优化的TCP Socket通信
- **负载均衡**: 智能任务分配算法
- **实时调整**: Token级动态重分配
- **性能优化**: 最小化端到端延迟

## 依赖项 / Dependencies

- PyTorch (模型推理)
- NumPy (数值计算)
- Socket (网络通信)
- Threading (并发处理)

## 注意事项 / Notes

1. 确保所有节点网络连通性
2. 模型文件路径正确性
3. 端口44444需要开放
4. 动态重分配功能需要性能追踪器支持