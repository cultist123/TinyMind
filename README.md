# TinyMind

从零手写的小型中文语言模型，用约 **26M 参数**理解 Transformer 与语言模型训练。

TinyMind 是一个面向学习与实验的 PyTorch 项目：从 RMSNorm、RoPE、分组查询注意力到自回归语言建模，逐步实现模型的核心组件，并提供预训练、文本生成、数据处理和 LoRA 适配器代码。代码中保留了较多中文注释，方便对照公式阅读与修改。

## 模型架构

模型采用 Decoder-only Transformer，默认配置如下：

| 配置 | 默认值 |
| --- | --- |
| 参数规模 | 约 26M，输入嵌入与输出层共享权重 |
| Transformer 层数 | 8 |
| 隐藏维度 | 512 |
| Query / KV 头数 | 8 / 2（GQA） |
| 词表大小 | 6,400 |
| 前馈网络 | SiLU 门控前馈网络 |
| 归一化 | RMSNorm，Pre-Norm |
| 位置编码 | RoPE，可选 YaRN 缩放配置 |
| 默认预训练序列长度 | 512 tokens |

注意力实现包含 PyTorch `scaled_dot_product_attention` 路径与显式注意力计算。模型通过 `PreTrainedModel` 和 `GenerationMixin` 接入 Transformers 的生成接口。

> 配置中的位置编码上限为 32,768，并不表示模型已经具备相应的长上下文能力；实际效果取决于训练数据、训练长度和训练结果。

## 当前进度

- **模型与预训练**：已提供核心模型、JSONL 数据加载、因果语言模型损失、梯度累积、梯度裁剪、学习率调度和检查点保存代码。预训练入口包含 CUDA、Apple MPS、CPU 设备选择与 CUDA DDP 支持。
- **文本生成**：`eval.py` 提供预设问题和交互输入两种模式，支持流式输出、温度采样与 top-p 采样。
- **数据处理**：包含预训练、SFT 和 DPO 数据集；SFT 仅为 assistant 回复构造训练标签，DPO 构造 chosen / rejected 对话及损失掩码。
- **LoRA**：已提供适配器注入、保存与加载工具，训练和推理入口仍需进一步整合。
- **待完善**：部分 SFT、DPO、LoRA、PPO 训练脚本仍引用旧的 `model.MokioModel`；SFT 数据返回值与部分训练入口也尚未统一。`train_grpo.py` 目前为空。MoE 仅有配置字段，当前模型尚未实现专家网络。

仓库目前未附带训练数据、模型权重或量化评测结果。以下命令基于现有代码入口整理，未代表已完成端到端训练验证。

## 目录结构

```text
TinyMind/
├── model/
│   ├── __init__.py          # TinyMindConfig
│   ├── TinyModel.py         # 注意力、前馈网络、Transformer 与语言模型
│   ├── model_lora.py        # LoRA 适配器工具
│   ├── tokenizer.json       # 分词器
│   └── tokenizer_config.json
├── dataset/
│   └── lm_dataset.py        # PretrainDataset / SFTDataset / DPODataset
├── trainer/
│   ├── train_pretrain.py    # 预训练入口
│   ├── trainer_utils.py     # 模型初始化、检查点、调度与分布式工具
│   └── train_*.py           # 后续训练阶段脚本（部分待适配）
├── torch_methods/          # PyTorch 基础操作学习示例
├── eval.py                 # 文本生成与交互入口
├── pyproject.toml          # Python 版本与依赖
└── uv.lock                 # 依赖锁文件
```

## 环境准备

项目要求 **Python 3.13 或更高版本**，主要依赖 PyTorch、Transformers、Datasets、NumPy 和 Pandas，具体版本见 [pyproject.toml](pyproject.toml)。

使用 uv 安装项目依赖：

```bash
git clone https://github.com/cultist123/TinyMind-clean.git
cd TinyMind-clean
uv sync
source .venv/bin/activate
```

NVIDIA GPU 环境需使用与本机驱动兼容的 PyTorch 构建。训练参数中的 `--use_wandb` 实际使用 SwanLab，若需要实验跟踪，可额外运行 `uv pip install swanlab`。

## 准备数据

预训练数据使用 JSONL 格式，每行包含一个 `text` 字段，例如：

```json
{"text": "语言模型通过预测下一个 token 学习文本中的规律。"}
{"text": "TinyMind 是一个用于学习 Transformer 的小型中文语言模型。"}
```

将自己的语料保存为 `dataset/pretrain.jsonl`。这些示例仅用于说明格式，有意义的训练需要足够规模和质量的语料。

SFT 数据集读取 `conversations` 字段：

```json
{"conversations": [{"role": "user", "content": "什么是语言模型？"}, {"role": "assistant", "content": "语言模型学习文本序列的概率分布。"}]}
```

DPO 数据集读取 `chosen` 和 `rejected` 字段，两者均为包含 `role`、`content` 的消息列表。

## 开始预训练

**从 `trainer/` 目录执行训练命令**，当前初始化工具默认按 `../model` 查找分词器，按 `../out` 查找权重。

```bash
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain.jsonl \
  --save_dir ../out \
  --from_weight none \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 512 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --epochs 1
```

预训练脚本默认优先选择 CUDA，其次 MPS，最后 CPU。可通过 `--device cuda:0`、`--device mps` 或 `--device cpu` 显式指定；显存或内存不足时可降低 `--batch_size` 或 `--max_seq_len`。

默认配置的输出权重为 `out/pretrain_512.pth`，可恢复训练的状态文件保存在 `checkpoints/`。在相同目录与配置下使用 `--from_resume 1` 尝试恢复已有检查点。

使用两张 NVIDIA GPU 的示例（同样在 `trainer/` 目录执行）：

```bash
torchrun --standalone --nproc_per_node=2 train_pretrain.py \
  --data_path ../dataset/pretrain.jsonl \
  --save_dir ../out \
  --from_weight none
```

## 文本生成

完成训练后，回到项目根目录运行：

```bash
cd ..
python eval.py \
  --load_from model \
  --save_dir out \
  --weight pretrain \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_new_tokens 128 \
  --device cpu
```

启动后选择 `0` 使用预设问题，选择 `1` 手动输入；手动模式输入空行结束。模型结构参数须与训练时保持一致。使用 NVIDIA GPU 时可将设备改为 `--device cuda`；当前推理脚本会将 MPS 请求回退为 CPU。

预训练权重主要用于文本续写，对话质量需要后续指令微调。当前 `eval.py` 的 LoRA 加载分支会抛出 `NotImplementedError`，暂时保持默认 `--lora_weight None`。

## 建议阅读顺序

1. `model/__init__.py`：了解模型配置与各维度。
2. `model/TinyModel.py`：沿 RMSNorm → RoPE → GQA → 前馈网络 → Transformer Block → Causal LM 阅读。
3. `dataset/lm_dataset.py`：理解 tokenization、padding、attention mask 与 labels 的关系。
4. `trainer/train_pretrain.py`、`trainer/trainer_utils.py`：串起训练循环、优化器与检查点。
5. `eval.py`、`model/model_lora.py`：继续研究自回归生成与参数高效微调。
