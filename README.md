# TinyMind

基于 [MiniMind](https://github.com/jingyaogong/minimind) 改编的轻量级中文语言模型，约 **26M 参数**，包含模型实现、数据处理、预训练与文本生成代码。

## 模型架构

8 层 Decoder，隐藏维度 512，8 个 Query 头共享 2 个 KV 头，词表大小 6400。输入 Embedding 和输出 LM Head 共享权重。

[![MiniMind 模型架构与数据流](https://raw.githubusercontent.com/cultist123/cultist123.github.io/main/dist/images/obsidian/54c927cb5358.png)](https://raw.githubusercontent.com/cultist123/cultist123.github.io/main/dist/images/obsidian/54c927cb5358.png)

模型实现在 [`model/TinyModel.py`](model/TinyModel.py)，参数在 [`model/__init__.py`](model/__init__.py)。

## 技术笔记

- [MiniMind：模型实现与张量数据流](https://cultist123.github.io/posts/minimind/)：GQA、FFN、残差连接与张量数据流。
- [RoPE 与 YaRN：位置编码与长度外推](https://cultist123.github.io/posts/rope-yarn/)：位置编码原理与 `precompute_freqs_cis`、`apply_rotary_pos_emb` 实现。
- [Transformer](https://cultist123.github.io/posts/transformer/)：注意力机制、多头注意力、掩码与前馈网络。

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
model/TinyModel.py         RMSNorm → RoPE → GQA → FFN → 完整模型
model/__init__.py          模型配置
dataset/lm_dataset.py      分词、padding、标签与 loss mask
trainer/train_pretrain.py  预训练循环
trainer/trainer_utils.py   初始化、学习率、检查点与分布式工具
eval.py                   加载权重与生成
model/model_lora.py        LoRA 适配器
torch_methods/            PyTorch 基础操作示例
```
