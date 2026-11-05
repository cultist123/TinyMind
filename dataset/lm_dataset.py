from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#数据增强
def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。
    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content



class PretrainDataset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length#输入给GPU的最大长度
        self.samples = load_dataset('json', data_files=data_path, split='train')    
    # __len__
    def __len__(self):
        return len(self.samples)

    # __getitem__
    # 我们拿到的是,jsonl里的每一行
    def __getitem__(self,index):
        sample=self.samples[index]

    # tokenizer会把文本转换成input_ids
        tokens = self.tokenizer(
            str(sample['text']),#这里假设每行jsonl都有一个'text'字段,包含了文本内容
            add_special_tokens=False,
            max_length=self.max_length-2,#留出位置给BOS和EOS
            truncation=True,#如果长度超过了max_length,会被截断  
        ).input_ids
    # 需要加上EOS,BOS,以及PAD填充
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))#填充到max_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)#转成tensor
    # 需要自行编写labels,防止PAD参与loss计算
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100#将PAD位置的label设为-100,防止参与loss计算
    # 需要编写attention_mask,告诉模型那些位置是有效的,哪些位置是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        #生成attention_mask,告诉模型哪些位置是有效的,哪些位置是PAD
        #非PAD位置为1,PAD位置为0
    # 我们要输出的,是input_ids和attention_mask,labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n",add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}",add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_input_ids(self, conversations):
        messages = [dict(m) for m in conversations]  # ！修正：逐条复制，下面要改tool_calls，别改到原数据
        tools=(
            conversations[0].get('tools')  # ！修正：缺少取值部分；数据集里tools字段名就叫'tools'
            if(
                conversations
                and conversations[0].get('role') =="system"  # ！修正：!= 改成 ==，tools只在system里
                and conversations[0].get('tools')
            ) else None
        )
        # ！修正：数据集里tools和tool_calls存的是JSON字符串，要转回list，否则chat_template会报错
        if isinstance(tools, str):
            tools = json.loads(tools)
        for m in messages:
            if isinstance(m.get('tool_calls'), str):
                m['tool_calls'] = json.loads(m['tool_calls'])
        # ！修正：原来缺少这一步和return，__getitem__拿到的是None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids):
        #让所有inputid都为-100
        #-100是交叉熵损失函数都默认忽略的值,也就是为-100不计算那部分
        labels = [-100]*len(input_ids)
        i=0
        #滑动窗口,寻找回答部分,并恢复原本的label
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start=i+len(self.bos_id)
                end=start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # ！修正：下面两句原来缩进在内层while里，应该和内层while平级
                for j in range(start, min(end+len(self.eos_id), len(input_ids))):
                    labels[j] = input_ids[j]
                i= end+len(self.eos_id) if end<len(input_ids) else len(input_ids)
            else:
                i += 1  # ！修正：没匹配到要往前走一格，否则死循环
        return labels
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        #是否要随机插入system_prompt
        conversations = pre_processing_chat(sample['conversations'])
        #用chat_template把对话转化成文本
        prompt=self.create_input_ids(conversations)
        #清空think块
        prompt = post_processing_chat(prompt)

        #toknize截断,并且补足pad
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] *(self.max_length - len(input_ids))
        #生成标签,只让assitant参与loss计算
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)



# ──────────────────────────────────────────────────────────────────────────────
# 3. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"偏好好回答、远离坏回答"，使输出更符合人类偏好
# 数据格式：{"chosen": [{role, content}...], "rejected": [{role, content}...]}
#   - chosen：人类标注的更优回答对话
#   - rejected：人类标注的较差回答对话
# 训练特点：
#   - 每条样本同时返回 chosen 和 rejected 两份 tokenized 序列，
#     训练时 DPO loss 会最大化 chosen 回复的对数似然、最小化 rejected 的。
#   - loss_mask 的设计与 SFT 一致：只有 assistant 回复部分为 1，
#     其余为 0，保证对比信号仅来自模型的实际输出部分。
#   - 采用"错位"方式构造输入输出对：x 取 [:-1]，y 取 [1:]，
#     即 x[t] 预测 y[t] = input[t+1]，标准自回归格式。
#   - mask 同样错位取 [1:]，与 y 对齐，方便在训练时直接做 masked loss。
#   - max_length 默认 4096，比 SFT 更长，因为 DPO 数据通常包含完整对话上下文。
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad_token_id 若不存在则回退到 0，保证补齐操作不会崩溃
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        # 与 SFTDataset 相同：预先 tokenize assistant 回复的起止标记，
        # 用于 generate_loss_mask 中精准定位 assistant 回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 优质回答对话列表，格式：[{role, content}, ...]
        rejected = sample["rejected"]  # 劣质回答对话列表，格式同上

        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标
        #         mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回的是具体的 token id（用于 CE loss）
        - DPO 返回的是 0/1 掩码（用于 masked 对数似然计算）
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1，其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
