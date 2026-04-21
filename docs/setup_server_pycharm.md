# 服务器与 PyCharm 远程开发环境

本文只说明本地/服务器开发环境，不涉及 CoordiWorld 的核心算法、真实数据读取、GPU 调用或训练流程。

## 1. 在服务器 clone repo

在服务器上选择一个代码目录，然后 clone 仓库：

```bash
git clone <repo-url> CoordiWorld
cd CoordiWorld
```

如果需要使用本轮建议分支，可以在 clone 后切换已有分支：

```bash
git switch infra/local-dev-env
```

如果需要本地新建分支：

```bash
git switch -c infra/local-dev-env
```

如果远端还没有该分支，则由维护者按团队流程创建；不要为了环境配置提交真实数据路径。

## 2. 创建 conda env

建议使用 Python 3.10 以上版本。示例：

```bash
conda create -n coordiworld python=3.10
conda activate coordiworld
python -m pip install --upgrade pip
```

## 3. 安装开发依赖

在仓库根目录执行：

```bash
python -m pip install -e ".[dev]"
```

该命令只安装 Python 包和开发工具，不会下载数据集，也不会调用 GPU。

## 4. 配置环境变量

仓库提供 `.env.example` 作为字段模板。可以把其中字段复制到服务器 shell、PyCharm Run/Debug Configuration，或私有 `.env` 文件中：

```bash
DATA_ROOT=/abs/path/to/datasets
NAVSIM_ROOT=/abs/path/to/navsim
OPENSCE_ROOT=/abs/path/to/opensce
NUSCENES_ROOT=/abs/path/to/nuscenes
WAYMO_ROOT=/abs/path/to/waymo
OUTPUT_ROOT=/abs/path/to/coordiworld_outputs
CHECKPOINT_ROOT=/abs/path/to/coordiworld_checkpoints
WANDB_MODE=offline
```

注意：

- 不要把真实数据路径提交到 git。
- 不要提交包含服务器绝对路径、用户名、挂载点或数据集位置的 `.env` 文件。
- `scripts/check_env.py` 只检查变量和目录可访问性，不会创建目录、下载数据或读取数据内容。
- `OPENSCE_ROOT` 是当前环境变量名；如果后续统一为 `OPENSCENE_ROOT`，需要单独开任务迁移。

如果使用 shell 临时加载私有 `.env`，可以在本机确认文件不会被提交后再执行：

```bash
set -a
source .env
set +a
```

## 5. 配置 PyCharm remote interpreter

在本机 PyCharm 中配置远程解释器：

1. 打开 `Settings | Project | Python Interpreter`。
2. 选择 `Add Interpreter | On SSH`。
3. 填写服务器 SSH 连接信息。
4. 选择服务器上的 conda 解释器，例如 `~/miniconda3/envs/coordiworld/bin/python`。
5. 确认 Project path mapping 指向服务器上的 `CoordiWorld` 仓库目录。
6. 在 PyCharm Terminal 中执行 `conda activate coordiworld` 后运行命令，或在 Run/Debug Configuration 的 Environment variables 中填写上述变量。

不要在 PyCharm 项目文件中保存真实数据路径后提交到 git。

## 6. pytest smoke test

安装完成后，可以先运行环境配置 smoke test：

```bash
pytest tests/test_env_config.py -q
python scripts/check_env.py --help
```

如果已经设置了环境变量并确认目录存在，可以再运行：

```bash
python scripts/check_env.py
```

该检查只访问路径元信息，不扫描真实数据目录内容。
