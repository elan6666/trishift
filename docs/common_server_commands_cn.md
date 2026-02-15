# TriShift 常用命令速查（本地 + 服务器）

## 1. 本地 PowerShell 常用

```powershell
# 连接服务器
ssh autodl-gpu

# 上传默认配置（覆盖服务器 defaults）
scp "E:\CODE\trishift\configs\defaults.yaml" autodl-gpu:/root/trishift/configs/defaults.yaml

# 上传某次实验配置为服务器默认配置
scp "E:\CODE\trishift\artifacts\results\adamson\state cond\defaults.yaml" autodl-gpu:/root/trishift/configs/defaults.yaml

# 下载 Adamson 结果目录
scp -r autodl-gpu:/root/trishift/artifacts/results/adamson "E:\CODE\trishift\artifacts\results\"

# 仅下载 csv/txt/pkl
scp autodl-gpu:/root/trishift/artifacts/results/adamson/*.csv "E:\CODE\trishift\artifacts\results\adamson\"
scp autodl-gpu:/root/trishift/artifacts/results/adamson/*.txt "E:\CODE\trishift\artifacts\results\adamson\"
scp autodl-gpu:/root/trishift/artifacts/results/adamson/*.pkl "E:\CODE\trishift\artifacts\results\adamson\"

# 在本地查看服务器配置前 40 行
ssh autodl-gpu "head -n 40 /root/trishift/configs/defaults.yaml"
```

## 2. 服务器终端常用（已登录后）

```bash
# 进入项目
cd /root/trishift

# 查看配置关键项
grep -nE '^train_mode:|^matching_mode:|n_splits:|num_workers:|pin_memory:' configs/defaults.yaml

# 运行 Adamson（前台 + 记录日志）
python -u scripts/run_adamson.py | tee run_adamson.log

# 检查本次实际训练模式
grep -n "\[run\] stage23 mode=" run_adamson.log

# 查看结果
ls -lh artifacts/results/adamson
cat artifacts/results/adamson/mean_pearson.txt
```

## 3. 服务器上快速改配置

```bash
cd /root/trishift

# 切 stage3_only
sed -i 's/^train_mode:.*/train_mode: stage3_only/' configs/defaults.yaml

# 切 matching_mode=ot
sed -i 's/^matching_mode:.*/matching_mode: ot/' configs/defaults.yaml

# 改 num_workers=4
sed -i 's/^\(\s*num_workers:\s*\).*/\14/' configs/defaults.yaml

# 查看是否改成功
grep -nE '^train_mode:|^matching_mode:|num_workers:' configs/defaults.yaml
```

## 4. 服务器 Git 更新

```bash
cd /root/trishift

# 强制同步远端最新（会丢弃服务器本地未提交改动）
git fetch origin && git reset --hard origin/main

# 查看最近提交
git log --oneline -n 5
```

## 5. 常见排查

```bash
# 看 GPU 是否可用
nvidia-smi

# 查看 Python / conda
python --version
conda --version

# 检查数据文件是否存在
ls -lh /root/trishift/src/data/adamson/perturb_processed.h5ad
```

## 6. 易错点

- `autodl-gpu` 是你本地 SSH 别名，只能在本地 PowerShell 用。
- 你已经在服务器里时，不要再执行 `ssh autodl-gpu ...`。
- `E:\...` 是 Windows 本地路径，不能在服务器终端直接使用。
