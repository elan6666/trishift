# TriShift 服务器训练教程（从零到跑通 Adamson）

本文档面向新手，目标是：  
1. 在服务器跑通 `scripts/run_adamson.py`  
2. 只上传 Adamson 必需数据  
3. 每条命令都说明“是干什么的”

---

## 1. 先确认本地 SSH 可用（本地 PowerShell）

```powershell
ssh autodl-gpu "echo ok"
```
作用：测试本地 SSH 别名 `autodl-gpu` 是否可连接。  
预期：输出 `ok`。

---

## 2. 服务器拉代码

```bash
ssh autodl-gpu
```
作用：登录服务器。

```bash
cd /root
```
作用：切换到服务器根目录下的工作区（常用放项目的位置）。

```bash
git clone https://github.com/elan6666/trishift.git
```
作用：把 GitHub 仓库下载到服务器。

```bash
cd trishift
```
作用：进入项目目录。

如果之前已经克隆过：

```bash
cd /root/trishift
git pull
```
作用：更新到仓库最新代码。

---

## 3. 只上传 Adamson 必需数据（本地 PowerShell 执行）

注意：`scp E:\...` 必须在 Windows 本地执行，不能在服务器终端执行。

```powershell
ssh autodl-gpu "mkdir -p /root/trishift/src/data/adamson /root/trishift/src/data/Data_GeneEmbd"
```
作用：先在服务器创建目标目录。`mkdir -p` 表示“目录不存在就创建，存在也不报错”。

```powershell
scp "E:\CODE\trishift\src\data\adamson\perturb_processed.h5ad" autodl-gpu:/root/trishift/src/data/adamson/
```
作用：上传 Adamson 数据文件。

```powershell
scp "E:\CODE\trishift\src\data\Data_GeneEmbd\GenePT_gene_embedding_ada_text.pickle" autodl-gpu:/root/trishift/src/data/Data_GeneEmbd/
```
作用：上传 Adamson 默认使用的 embedding（`emb_b`）。

---

## 4. 改路径配置为 Linux 路径（服务器执行）

```bash
ssh autodl-gpu
cd /root/trishift
```
作用：回到服务器并进入项目目录。

```bash
sed -i 's#adamson:.*#adamson: "/root/trishift/src/data/adamson/perturb_processed.h5ad"#' configs/paths.yaml
```
作用：把 `paths.yaml` 里 Adamson 路径替换为服务器路径。  
说明：`sed -i` 是“原地修改文件”。

```bash
sed -i 's#emb_b:.*#emb_b: "/root/trishift/src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle"#' configs/paths.yaml
```
作用：把 `emb_b` 路径替换为服务器路径。

```bash
cat configs/paths.yaml
```
作用：打印文件全文，检查刚才改动是否生效。  
你问的 `cat`：它就是“把文件内容输出到终端”。

---

## 5. 创建并激活 Conda 环境

```bash
conda create -n scouter python=3.10 -y
```
作用：创建名为 `scouter` 的 Python 3.10 环境。`-y` 表示自动确认。

```bash
conda init bash
source ~/.bashrc
```
作用：初始化 conda shell，并重新加载 shell 配置，解决 `conda activate` 报错。

```bash
conda activate scouter
```
作用：进入 `scouter` 环境。

---

## 6. 安装依赖

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
作用：安装 GPU 版 PyTorch（CUDA 12.1 对应包）。

```bash
pip install -e .
```
作用：可编辑安装当前项目，代码改动会立即生效。

```bash
pip install scanpy anndata pandas scipy scikit-learn tqdm pyyaml matplotlib seaborn
```
作用：安装训练/评估常用依赖。

---

## 7. 验证环境

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```
作用：检查 torch 版本、CUDA 是否可用。

```bash
nvidia-smi
```
作用：检查 GPU 设备状态和显存占用。

---

## 8. 推荐性能设置（可选）

```bash
sed -i 's/^[[:space:]]*num_workers:.*/  num_workers: 4/' configs/defaults.yaml
sed -i 's/^[[:space:]]*pin_memory:.*/  pin_memory: true/' configs/defaults.yaml
grep -n -E 'num_workers|pin_memory' configs/defaults.yaml
```
作用：设置 DataLoader 并验证。  
`grep -n`：按关键字搜索并显示行号；`-E`：启用正则表达式。

---

## 9. 启动训练

```bash
cd /root/trishift
conda activate scouter
python -u scripts/run_adamson.py | tee run_adamson.log
```
作用：开始训练并记录日志。  
- `-u`：Python 不缓冲输出，日志实时显示  
- `tee run_adamson.log`：屏幕显示的同时写入日志文件

---

## 10. 结果查看

```bash
tail -n 50 run_adamson.log
```
作用：查看日志最后 50 行，快速看是否报错。  
`tail` = 看文件末尾。

```bash
cat artifacts/results/adamson/mean_pearson.txt
```
作用：查看平均指标摘要。

```bash
head -n 20 artifacts/results/adamson/metrics.csv
```
作用：查看指标表前 20 行。  
`head` = 看文件开头。

---

## 11. 防断线运行（推荐）

```bash
apt update && apt install -y tmux
```
作用：安装 `tmux`，断开 SSH 后训练也能继续。

```bash
tmux new -s adamson
```
作用：创建名为 `adamson` 的 tmux 会话。

```bash
conda activate scouter
cd /root/trishift
python -u scripts/run_adamson.py | tee run_adamson.log
```
作用：在 tmux 里训练。

快捷键：  
- `Ctrl+B` 再按 `D`：退出会话但任务继续  
- `tmux attach -t adamson`：重新进入会话

---

## 12. 常见报错处理

### 12.1 `ModuleNotFoundError: No module named 'torch'`

```bash
conda run -n scouter pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n scouter python -u scripts/run_adamson.py | tee run_adamson.log
```
作用：不依赖 `conda activate`，直接指定环境安装并运行。

### 12.2 `CondaError: Run 'conda init' before 'conda activate'`

```bash
conda init bash
source ~/.bashrc
conda activate scouter
```
作用：初始化 conda shell。

### 12.3 `nano: command not found`

说明：服务器没有 nano。  
处理：优先用 `sed -i` 修改配置（上文已给）。

### 12.4 `scp: Could not resolve hostname e`

原因：在服务器里执行了 Windows 路径命令。  
处理：回到本地 PowerShell 执行 `scp`。

---

## 13. 命令词典（你问到的 `cat` 在这里）

- `cat 文件`：打印文件全文到终端。常用于“检查配置文件内容”。  
- `grep 关键词 文件`：在文件中搜索关键词。  
- `grep -n`：显示匹配行的行号。  
- `head -n N 文件`：显示文件前 N 行。  
- `tail -n N 文件`：显示文件后 N 行。  
- `tee 文件`：把终端输出“同时”保存到文件。  
- `sed -i 's/旧/新/' 文件`：直接在文件中替换文本。  
- `cd 目录`：切换目录。  
- `ls -lh`：列出文件并显示人类可读大小。  
- `mkdir -p 路径`：递归创建目录，已存在不报错。  
- `scp 本地 远端`：本地与远端之间传文件。  
- `ssh 主机`：登录远程服务器。  
- `conda create -n 名称 python=版本`：创建环境。  
- `conda activate 名称`：激活环境。  
- `conda run -n 名称 命令`：不激活也可在指定环境运行命令。  
- `nvidia-smi`：查看 GPU 状态。  
- `tmux`：断线不中断任务的会话管理工具。

