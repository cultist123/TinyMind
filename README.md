# TinyMind

基于 [MiniMind](https://github.com/jingyaogong/minimind) 改编的轻量级中文语言模型，约 **26M 参数**，包含模型实现、数据处理、预训练与文本生成代码。

## 模型架构

8 层 Decoder，隐藏维度 512，8 个 Query 头共享 2 个 KV 头，词表大小 6400。输入 Embedding 和输出 LM Head 共享权重。

[![MiniMind 模型架构与数据流](https://raw.githubusercontent.com/cultist123/cultist123.github.io/main/dist/images/obsidian/54c927cb5358.png)](https://raw.githubusercontent.com/cultist123/cultist123.github.io/main/dist/images/obsidian/54c927cb5358.png)

模型实现在 [`model/TinyModel.py`](model/TinyModel.py)，参数在 [`model/__init__.py`](model/__init__.py)。

## 技术笔记

相关原理、公式推导与实现记录整理在[个人博客](https://cultist123.github.io/)。

### 模型基础

- [序列模型：N 元语法、RNN 与困惑度](https://cultist123.github.io/posts/sequence-models/)
- [Transformer](https://cultist123.github.io/posts/transformer/)
- [GPT 与 BERT：预训练、微调与 Prompt](https://cultist123.github.io/posts/gpt-bert/)
- [GPT-3 与 GPT-4：Few-shot 与模型演进](https://cultist123.github.io/posts/gpt3-gpt4/)

### 模型实现

- [MiniMind：模型实现与张量数据流](https://cultist123.github.io/posts/minimind/)
- [LayerNorm 与 RMSNorm：原理与实现](https://cultist123.github.io/posts/rmsnorm/)
- [RoPE 与 YaRN：位置编码与长度外推](https://cultist123.github.io/posts/rope-yarn/)
- [Padding Mask：填充掩码与因果掩码](https://cultist123.github.io/posts/padding-mask/)

### 指令微调与 LoRA

- [SFT：指令格式与回答部分的 Loss](https://cultist123.github.io/posts/sft/)
- [LoRA](https://cultist123.github.io/posts/lora/)

### 强化学习与对齐

- [强化学习基础：MDP、策略与回报](https://cultist123.github.io/posts/mdp/)
- [Bellman 方程：价值迭代与策略迭代](https://cultist123.github.io/posts/bellman/)
- [Policy Gradient：推导、Baseline 与训练流程](https://cultist123.github.io/posts/policy-gradient/)
- [Actor-Critic：从策略梯度到 A2C 与 A3C](https://cultist123.github.io/posts/actor-critic/)
- [PPO：重要性采样、约束与 Clip](https://cultist123.github.io/posts/ppo/)
- [RLHF-PPO：四个模型、GAE 与训练代码](https://cultist123.github.io/posts/rlhf-ppo/)
- [GRPO：目标函数与组内优势估计](https://cultist123.github.io/posts/grpo/)

## 使用方法

需要 Python 3.13+，依赖见 [`pyproject.toml`](pyproject.toml)。使用 uv：

```bash
git clone https://github.com/cultist123/TinyMind.git
cd TinyMind
uv sync
source .venv/bin/activate
```

准备 `dataset/pretrain.jsonl`，每行一个 `text` 字段：

```json
{"text": "这里放一段训练文本。"}
{"text": "每行是一条样本，训练时会分词、截断，并补齐到指定长度。"}
```

训练命令需在 `trainer/` 目录执行，分词器和权重路径相对于该目录解析：

```bash
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain.jsonl \
  --save_dir ../out \
  --from_weight none \
  --batch_size 4 \
  --max_seq_len 512 \
  --epochs 1
```

脚本会优先选择 CUDA，其次 MPS，最后 CPU，也可以用 `--device` 指定。内存不足时可减小 batch size 或序列长度。默认权重存到 `out/pretrain_512.pth`，恢复训练的状态存在 `checkpoints/`，续训参数是 `--from_resume 1`。

训练完成后，在项目根目录执行文本生成：

```bash
cd ..
python eval.py --weight pretrain --max_new_tokens 128 --device cpu
```

选择 `0` 使用预设问题，选择 `1` 进入交互模式，空行退出。使用 NVIDIA GPU 时可指定 `--device cuda`；当前推理入口遇到 MPS 会回退到 CPU。预训练权重用于文本续写，对话任务需进一步指令微调。

## 代码结构

```text
TinyMind/
├── model/
│   ├── __init__.py          # 模型配置
│   ├── TinyModel.py         # RMSNorm、RoPE、GQA、FFN 与完整模型
│   ├── model_lora.py        # LoRA 适配器
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── merhod/              # 注意力与归一化模块练习
│       ├── gqa.py
│       ├── rmsnorm.py
│       └── rpoe.py
├── dataset/
│   ├── __init__.py
│   └── lm_dataset.py        # 预训练、SFT 与 DPO 数据处理
├── trainer/
│   ├── train_pretrain.py    # 预训练
│   ├── train_full_sft.py    # 全参数微调
│   ├── train_lora.py        # LoRA 微调
│   ├── train_dpo.py         # DPO
│   ├── train_ppo.py         # PPO
│   ├── train_grpo.py        # GRPO 脚本占位
│   └── trainer_utils.py     # 初始化、学习率、检查点与分布式工具
├── torch_methods/          # PyTorch 基础操作练习
├── src/
│   └── minimind/
│       └── __init__.py
├── eval.py                 # 权重加载与文本生成
├── main.py
├── pyproject.toml          # 项目配置与依赖
├── uv.lock
└── README.md
```
