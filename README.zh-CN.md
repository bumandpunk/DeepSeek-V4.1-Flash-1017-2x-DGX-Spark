# Spark1017

在两台 DGX Spark 上运行 DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw，提供代码编辑加速和双机部署脚本。

`v0.2.0-alpha.1` · [English](README.md) · [部署文档](docs/operations.md) · [测试结果](results/release-v0.2.0-alpha.1/README.md)

<!-- validation-status:start -->
Alpha：新缓存部署后的连续 128K 输入仍有内存分配问题。[问题与测试记录](docs/known-issues.md)
<!-- validation-status:end -->

基于 [TensorFold](https://github.com/bertholomus/TensorFold)，主要增加了：

- **代码编辑加速**：从上下文中寻找可复用的代码，由主模型批量验证，减少重复生成。
- **Engram 分块归约**：缩小预填充阶段的临时缓冲。在测试尺寸下，额外峰值分配从 600 MiB 降至 132 MiB，两台 GB10 的逐位一致检查均通过。
- **实验性通信融合**：将分段 RDMA 接收接入 HC 残差计算。
- **双机管理**：统一构建、预编译、启动、状态检查、停机和原服务恢复。

## 硬件与环境

- 两台 DGX Spark / GB10，通过双路 ConnectX-7/RoCE 连接。网络配置见 NVIDIA 的 [双 Spark 连接指南](https://build.nvidia.com/spark/connect-two-sparks)。
- 两节点安装 Docker、NVIDIA 容器运行时和 Python 3.11+；主节点可免交互 SSH 到从节点，systemd 用户服务可用。
- 两节点各自存放模型和 Engram 文件，来源见 [权重说明](checkpoints/README.md)。权重需另行下载。
- 每节点另留至少 130 GiB 缓存空间；启动前至少有 110 GiB 可用内存。

默认两个槽位共享 256K token 窗口，长输入测试覆盖 128K 输入加 512 token 输出。512K 冷启动验证未通过。

RTX 4090 不是运行依赖。我们的测试使用它所在的主机执行测量、CI 和隔离评分。

## 快速开始

在主节点执行：

```bash
git clone https://github.com/bumandpunk/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark.git
cd DeepSeek-V4.1-Flash-1017-2x-DGX-Spark
cp config/cluster.example.json config/cluster.local.json
```

编辑 `config/cluster.local.json`，填写两节点地址、网卡/GID、模型和 Engram 路径，以及各自独立的缓存、状态目录。目录应由部署用户拥有。

API 密钥放在仓库外，两节点使用相同密钥，文件权限设为 `0600`。配置示例中的 `key_file` 指向这个文件。

```bash
python3 scripts/cluster.py doctor
python3 scripts/build.py --no-cache
python3 scripts/cluster.py prebuild
./start.sh --profile edit
./verify.sh
```

构建脚本会固定上游版本，并把同一镜像分发到从节点。首次启动需要生成编译和 rank 缓存，后续启动可复用。

国内网络可在构建时指定清华镜像，下载包仍按固定的官方 SHA-256 校验：

```bash
python3 scripts/build.py --no-cache --package-index https://pypi.tuna.tsinghua.edu.cn/simple
```

默认 API 地址为 `http://127.0.0.1:19890/v1`，模型名为 `DeepSeek-V4.1-Flash-1017`，请求需要 API 密钥。

启动时默认设置两小时自动恢复计时器。确认要持续运行后，执行：

```bash
python3 scripts/cluster.py confirm
```

## 运行配置

| 配置 | 用途 |
|---|---|
| `baseline` | 关闭代码候选和通信融合，作为对照 |
| `edit` | 开启重复代码候选，适合保留大量原代码的编辑任务 |
| `prefill-experimental` | 单独启用 RDMA/HC 融合，用于长输入实验 |

切换配置前先执行 `./stop.sh`。代码候选与通信融合暂不组合启用。

| 操作 | 命令 |
|---|---|
| 查看双机状态 | `./status.sh` |
| 检查 API | `./verify.sh` |
| 停止当前实例 | `./stop.sh` |
| 恢复原服务 | `python3 scripts/cluster.py restore` |

接管已有部署时，先按 [恢复配置](docs/operations.md) 填写原服务、健康检查、文件和容器列表，再执行 `./start.sh --profile edit --takeover`。接管前会保存快照；恢复后核对原文件、容器规格和 API。

## 测试结果

目前的速度收益主要出现在保留大量已有代码的完整文件编辑。普通生成和长输入还没有稳定的速度优势。

- [完整文件编辑、CanItEdit 和 SciCode](results/2026-10-27/README.md)
- [普通生成、并发和长上下文](results/2026-10-28/README.md)
- [本版本部署与回归记录](results/release-v0.2.0-alpha.1/README.md)
- [对照项目与测试条件](benchmarks/competitors.md)

这些目录包含运行配置、原始响应、功能测试和失败记录。修改代码后可先运行 CPU 检查：

```bash
bash scripts/ci.sh
```

## 开发与许可

[代码候选设计](docs/design.md) · [通信融合](docs/round2-design.md) · [Engram 工作空间](docs/engram-reduction.md) · [贡献指南](CONTRIBUTING.md)

新代码采用 Apache-2.0。TensorFold 提供模型执行、EXL3 内核、张量并行和 DSpark；第三方代码与测试集的许可见 [NOTICE](NOTICE) 和 [LICENSES](LICENSES/)。
