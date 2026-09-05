from .msrnn       import MsRNN
from .transformer  import CausalTransformer
from .ttt_rnn      import TTTLinear
from .deltanet     import GatedDeltaNet

MODEL_REGISTRY = {
    "msrnn":       MsRNN,
    "transformer": CausalTransformer,
    "ttt":         TTTLinear,
    "deltanet":    GatedDeltaNet,
}


def get_model(model_cfg, train_cfg):
    name = model_cfg.model
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](model_cfg, train_cfg)
