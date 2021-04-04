#added progress flag to model getters 向模型获取器添加进度标志
try:
    from torch.hub import load_state_dict_from_url
    # pytorch hub 是一个预训练模型库（包括模型定义和预训练权重）
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url