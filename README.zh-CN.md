# Spark1017

<!-- validation-status:start -->
**源码预览状态：** 最终镜像的完整编程回归与结束恢复正在运行。[已完成检查和待完成项](results/release-v0.2.0-alpha.1/README.md) 单独记录；CPU CI 通过不代表尚未完成的 GPU 验证已经通过。
<!-- validation-status:end -->

**面向两台 NVIDIA DGX Spark / GB10 的代码编辑加速与推理优化实验。**

`v0.2.0-alpha.1` · Apache-2.0 · 开发者预览版 · [English](README.md)

项目在固定版本的 TensorFold 上增加重复代码候选、实验性 RDMA/HC 融合和可复现的双机管理流程。模型执行、EXL3 数值内核、张量并行和 DSpark 来自上游。当前改造的是推理系统，尚未训练新的基础模型。

[快速开始](#快速开始) · [运行及恢复](docs/operations.md) · [实现设计](docs/design.md) · [发布说明](docs/releases/v0.2.0-alpha.1.md) · [贡献指南](CONTRIBUTING.md)

## 已测优势

同一对 GB10、同一套 A0 EXL3 2.9 bpw 权重、相同输入与输出预算：

下表是 2026-10-27/28 双方均使用 512K 共享窗口的历史对照。本次发布验证改用 256K 窗口，单独记录 baseline、edit 和通信配置的结果；没有在这一容量下重跑 Mia。

| 场景 | Mia 原生引擎 | Spark1017 | 结果 |
|---|---:|---:|---|
| bisect 完整文件编辑 | 112.35 tok/s | 148.41 tok/s | 解码 +32.1%，整次耗时 −23.1% |
| graphlib 完整文件编辑 | 120.06 tok/s | 189.18 tok/s | 解码 +57.6%，整次耗时 −28.9% |
| textwrap 完整文件编辑 | 120.06 tok/s | 188.35 tok/s | 解码 +56.9%，整次耗时 −34.1% |
| 普通短代码生成 | 96.32 tok/s | 95.08 tok/s | Mia 更快 |
| 128K 输入处理 | 2341.60 input tok/s | 2047.74 input tok/s | Mia 更快 |

编辑行使用第一轮 lookup；普通生成和长输入行使用第二轮通信融合。这些不是同一个组合配置。三个编辑任务每题重复两次，完整输出一致，每个引擎通过 836 次功能断言。样本少、同日顺序运行，不能外推为所有开发任务都提速。

冻结的 E320 编程回归保持 SciCode **37/50**，CanItEdit 两次均 **7/12**，配对代码和判定一致。SciCode 参考答案控制只通过 48/50；这是开发回归，不是 Artificial Analysis 官方成绩，也不证明模型能力增强。

[第一轮原始结果](results/2026-10-27/README.md) · [第二轮原始结果](results/2026-10-28/README.md) · [对照条件](benchmarks/competitors.md)

## 实现与适用范围

重复代码索引从已提交的上下文寻找候选，再由主模型批量验证。收益主要来自保留大量已有代码的完整文件或代码块编辑；短补丁、全新代码和多请求并发的收益可能更小。多个活跃流时会回退到原 DSpark 调度。

另一条独立路线将分段 RDMA 接收直接接到 HC 残差处理，减少中间张量读写。它相对 Python 基线有小幅长输入收益，但同条件下仍落后于已测 Mia 原生引擎。失败实验、对照和来源均保留，不宣称所有场景最快。

Engram 归约采用确定的按行分块通信，把两卡 FP32 部分结果直接相加到 BF16 输出切片。代表性预填充尺寸的额外峰值分配在两台 GB10 上均从 600 MiB 降至 132 MiB，逐位一致检查通过。这是工作空间节省，不单独作为端到端提速结论。分块和推测解码已有先例，本项目贡献是面向这一模型与双机环境的实现、集成及实测，详见 [Engram 工作空间设计](docs/engram-reduction.md)。

## 准备环境

- 两台 DGX Spark / GB10，已配置 Docker、NVIDIA 运行时和双路 ConnectX-7/RoCE。先按 NVIDIA 的 [双 Spark 连接指南](https://build.nvidia.com/spark/connect-two-sparks) 配置网络和 SSH。
- 两台机器都能从本地读取兼容的模型与 Engram 文件。固定来源见 [checkpoints/sources.json](checkpoints/sources.json)。模型不随项目分发。
- 除模型、Engram、镜像外，每节点为全新缓存额外预留至少 130 GiB 磁盘。示例启动门槛为 110 GiB 可用内存，运行时保留 2.5 GiB 主机内存保护，引擎缓存预算另留 0.5 GiB 余量（默认目标为 3.0 GiB）。
- 发布默认值为两个槽位共享 256K token 窗口，长输入回归覆盖 128K 输入加 512 token 输出。512K 冷启动曾因安全工作空间不足被拒绝，不属于本版验证通过的配置。
- 两节点 Python 3.11+，主节点到从节点的免交互 SSH，以及用于定时恢复的 systemd 用户管理器。
- 4090 可选。本项目实测时，它的主机负责测量、Linux CI 和隔离评分，不参与最终主模型推理。

## 快速开始

在主节点的源码目录执行：

```bash
cp config/cluster.example.json config/cluster.local.json
```

填好地址、网卡/GID、模型路径以及独立的缓存和状态目录；目录应由部署用户拥有。模型路径要直接指向包含 `config.json` 和权重索引的目录。API 密钥放在仓库外，两节点使用相同密钥，权限设为 0600。默认只监听本机地址。

```bash
python3 scripts/cluster.py doctor
python3 scripts/build.py --no-cache
python3 scripts/cluster.py prebuild
./start.sh --profile edit
./verify.sh
./status.sh
```

构建脚本验证官方基础镜像 ID，取得固定上游提交、应用公开补丁，并将同一个新镜像分发到从节点。原生扩展在空闲 GPU 上预编译，限制单编译任务和 8 GiB 内存。服务使用独立的缓存目录，首次启动建立 rank cache，之后可复用；每次启动使用新的 prompt spill 目录。

国内网络可使用 `python3 scripts/build.py --no-cache --package-index https://pypi.tuna.tsinghua.edu.cn/simple`。镜像下载的安装包仍须通过固定的官方 PyPI SHA-256 校验；不会修改系统的 pip 配置。

默认 API 为 `http://127.0.0.1:19890/v1`，模型名 `DeepSeek-V4.1-Flash-1017`，必须提供密钥。启动会等待认证就绪并完成五项真实 API 检查。

启动前会设置自动恢复期限；确认要长期保留实验服务时，执行 `python3 scripts/cluster.py confirm` 重新检查 API 并解除定时恢复，内存保护仍然有效。

## 接管和恢复原服务

在配置中填写 `previous_service`、`previous_health`，以及两节点的 `previous_paths` 和 `previous_containers`。

```bash
python3 scripts/cluster.py snapshot
./start.sh --profile edit --takeover
```

快照保存原配置、哈希、容器规格和镜像标签。接管先设置独立恢复定时器，再停止明确配置的原服务；不会停止不相关容器。

```bash
./stop.sh
python3 scripts/cluster.py restore
```

`stop` 只停止本项目容器。`restore` 会检查备份完整性和原文件是否被修改，恢复此前运行的原服务，核对容器规格并测试 API，成功后才解除恢复定时器。完整流程、日志和异常处理见 [运行及恢复](docs/operations.md)。

## 三种配置

| 配置 | 重复代码 lookup | RDMA/HC 融合 |
|---|---|---|
| baseline | 关闭 | 关闭 |
| edit | 开启 | 关闭 |
| prefill-experimental | 关闭 | 开启 |

切换前先停止当前实例。没有默认组合尚未验证的两种优化；其他实验路线全部关闭。

## 复核与贡献

```bash
bash scripts/ci.sh
python3 results/release-v0.2.0-alpha.1/check.py
```

CI 包含 CPU 不变量、生命周期故障路径、历史结果一致性和原生缓存的 ASan/UBSan 检查。它不代表每个提交都做了双机 GPU/RDMA 实测。发布前的干净部署记录见 [发布说明](docs/releases/v0.2.0-alpha.1.md)。

保留上游 encoder prefill 加有界 decoder replay 的行为，不把它描述成完整精确 decoder prefill。没有提供长时间生产稳定性、RDMA 断链恢复或所有长上下文编程任务的保证。E320 是专家剪枝变换，改变了模型函数，不能称为无损压缩。

项目结构参考 MiaAI-Lab、Elevatormusic 等公开双 Spark 项目的状态说明、快速开始和运维组织方式。新的实现采用 Apache-2.0，第三方归属见 [NOTICE](NOTICE)。欢迎按 [贡献指南](CONTRIBUTING.md) 提交可复现的改进和负结果。

Engram 两卡求和与 BF16 转换的合并、逐位验证及峰值内存测量见 [Engram 工作区说明](docs/engram-reduction.md)。这项算子测量不等同于端到端提速或完整质量证明。
