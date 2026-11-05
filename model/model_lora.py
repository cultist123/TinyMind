# LoRA 微调旁支
import torch
import torch.nn as nn
import torch.optim as optim

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0, std=0.02)
        # 初始化 B 矩阵为全零，确保初始状态下W‘=W,得到最初的输出再优化
        self.B.weight.data.normal_(mean=0, std=0)

    def forward(self, x):
        return self.B(self.A(x))#先算A再算B

#模型嫁接,把LoRA嫁接到模型上
def apply_lora(model,rank=8):
    for name,module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora=LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)

            setattr(module, "lora", lora)#打成标志
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward,layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora

def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)

