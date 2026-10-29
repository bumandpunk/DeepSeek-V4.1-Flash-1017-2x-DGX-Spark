# Spark1017

面向两台 DGX Spark / GB10 的 DeepSeek-V4.1-Flash-1017 推理优化实验。项目提供独立的分段通信/HC 融合、存储、候选生成和验证调度代码，以及固定输入的实机对照。模型执行、EXL3 CUDA 内核、RDMA 和 DSpark 来自 TensorFold；这是推理系统改造，尚未训练新的基础模型。

## 实现

- **分段 RDMA 接收与 HC 后处理融合**：约 51 MiB 注册环、双段流水线与 GPU 序号 ticket；按原 rank 顺序和 BF16 舍入边界直接更新残差，减少中间张量读写。
- **重复代码候选与 DSpark 回退**：有界后缀索引寻找已经出现过的代码片段，以最多 15 个候选填充主模型验证窗口；根据接受率和成本决定是否使用候选。并发请求回退到原 DSpark 调度。
- **分区驻留与精确行缓存**：只驻留本 rank 会读取的 Engram 缩放表；有界热点缓存与批内重复读合并保留原始字节，并将统一内存预算纳入启动检查。
- **接受率校准的验证调度**：按深度/置信度记录真正到达的验证位置，用动态规划分配验证行数。两台机器使用同一固定成本表与确定性状态更新。

这些模块可以独立开关。上下文查找、缓存与推测解码都有公开先例；本项目的贡献范围、状态约束及引用见 [设计说明](docs/design.md)。没有“全球首创”或“所有场景最快”的声明。

<!-- ROUND2_RESULTS_START -->
## 第二轮：普通生成与长上下文（2026-10-28）

新增分段 RDMA 接收与 HC 后处理融合。相对同权重 Python 基线，128K 预填充提升 **E320 3.03% / A0 4.52%**。普通生成没有稳定提速：A0 代码生成回切基线 94.81 tok/s，融合版 95.08（+0.28%）。

同 A0 权重的 Mia 原生方案仍更快：普通代码 **96.32 vs 95.08 tok/s**，128K 预填充 **2341.6 vs 2047.7 tok/s**。三个引擎的 58 个配对响应完全相同；本项目尚未达到通用场景更快的目标。

SciCode 回归仍为 **37/50**，50 个代码和判定相同；CanItEdit 两次均为 **7/12**，24 个文本、代码和判定相同。固定样本回归不等于 AA 官方成绩或普遍质量无损。

[完整结果、原始记录与边界](results/2026-10-28/README.md) · [第二轮设计](docs/round2-design.md) · [变更记录](CHANGELOG.md)

```bash
python3 tools/summarize_round2.py
python3 tools/check_round2.py
```

微基准约 27% 的局部时间减少不作为整机收益。4090 远程草稿等未获益路线保留源码与负结果，正式候选关闭。4090 在最终对照中承担公共客户端和隔离评分。
<!-- ROUND2_RESULTS_END -->

## 第一轮：保留原源码的编辑任务（2026-10-27）

|同条件对照|结果|边界|
|---|---|---|
|同 A0 权重：Mia 原生方案 → Spark1017|3 个完整源码编辑任务解码提升 **32–58%**，整次耗时下降 **23–34%**|每题 2 次，836 次功能断言/引擎，配对完整输出相同；保留大量原源码的任务|
|同 E320 权重：lookup 关闭 → 开启|相同 3 个完整源码编辑任务解码提升 **42–80%**|请求、完整输出、token 数相同|
|SciCode 开发回归|**37/50 → 37/50**，50 个抽取代码哈希相同|固定协议、小开发集；参考答案控制仅 48/50，不是 AA 官方分数|
|CanItEdit 固定 12 题子集|两次重复均 **7/12 → 7/12**，全部配对文本相同|7 道正确题中 6 道解码提升 24–48%，1 道下降约 7%；保留失败题|

普通短代码生成未表现出明显通用提速。存储与校准调度仍为默认关闭的实验模块。本节为第一轮历史结果，以上为同日顺序测量，不构成跨日统计显著、普遍质量无损或全部开源方案排名。完整协议、长输入、并发、失败记录和原始请求见 [实机结果](results/2026-10-27/README.md)。

[开源引擎对照设置](benchmarks/competitors.md) 记录源码提交、公开补丁和实际启动差异。可运行 `python3 tools/check_results.py` 核对所附原始记录与结果表的一致性；它不会执行生成的代码，也不能替代重新进行实机测量。

## 复现基础

实机配置为两台 GB10、每台约 121.69 GiB 可见统一内存、双 200G CX7 连接。4090 主机承担测量客户端、Linux 扩展校验和隔离代码评分。4090 不参与主模型推理。实测环境的内核、镜像、模型、上下文、并发和请求哈希需与结果表一并阅读。

CPU 不变量测试无需模型或 GPU：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_algorithms.py -v
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -pthread tests/test_tiered_store.cpp -o /tmp/spark1017-test
/tmp/spark1017-test
```

Linux AIO 扩展测试需要 Linux、PyTorch C++ 扩展工具链和支持 O_DIRECT 的本地文件系统：

```bash
MAX_JOBS=1 python3 tests/test_linux_aio.py
```

构建来源固定为 TensorFold `07aa02585f64cfb04c60a96de6554c0d4708ed46`。`compat/deployment-safety.patch` 保留实测部署的内存、取消请求和 spill 修复；`tools/prepare_source.py` 校验原始哈希后应用补丁和新模块。完整源码重建已与实机加载文件逐字节比对。Dockerfile 是构建配方；实机验证使用相同基础镜像加这些源文件挂载，尚未把该 Dockerfile 的全新镜像构建当作已验证结果。

```bash
docker build -t spark1017:experiment .
```

先停止推理再构建扩展，并限制编译并行与内存。模型、Engram 和 rank cache 应独立挂载，模型只读；保持两台机器的源码、权重、环境变量和成本文件一致。运行参数与各模块开关见 [运行说明](docs/runtime.md)。

## 模型和验证边界

A0 是现有 1017 EXL3 2.9 bpw 的原样副本。可选 E320 用 [导出工具和选择表](checkpoints/README.md) 将 384 个路由专家保留为 320 个，保留的量化张量字节不变。剪枝仍会改变模型函数，不能称为无损模型压缩。

短提示速度、完整代码编辑、长输入预填充、并发吞吐和编程正确性分别报告。SciCode 开发集结果不等同于 Artificial Analysis 排行榜，也不能和不同协议的官方分数直接比较。速度测试题不进入模型训练；每次生成的代码仅在无网络、只读、资源受限的容器中评分。

冻结的编程回归协议、数据来源、执行容器和可移植脚本见 [SciCode 回归说明](benchmarks/scicode/README.md) 和 [CanItEdit 子集](benchmarks/canitedit/README.md)。完整源码编辑的独立功能断言见 [编辑测试说明](benchmarks/edit/README.md)。

## 来源和许可证

[TensorFold](https://github.com/bertholomus/TensorFold)、[lucataco 双机部署](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark)、[MiaAI 原生方案](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) 和 [sudoingX vLLM 方案](https://github.com/sudoingX/dgx-spark-deepseek-v41) 是研究与比较对象。保留的第三方声明位于 [NOTICE](NOTICE) 和 LICENSES/。新增代码按 Apache-2.0 提供；不分发模型权重、缓存、密钥或私人部署配置。
