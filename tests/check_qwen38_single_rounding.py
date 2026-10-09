import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL

config = Qwen4ExpConfig(text_config=dict(
    vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
    linear_key_head_dim=16, linear_value_head_dim=16, num_experts=2, num_experts_per_tok=1,
    moe_intermediate_size=16, shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=32, indexer_compress_ratio=4,
    ple_layer_ids=[], layer_types=['linear_attention', 'full_attention']),
    vision_config=dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=2,
                       out_hidden_size=32, deepstack_visual_indexes=[]))
with torch.device('meta'):
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)
norm = model.model.language_model.layers[0].linear_attn.norm.to_empty(device='cpu').bfloat16()
with torch.no_grad():
    norm.weight.fill_(0.953125)
torch.manual_seed(17)
x, gate = torch.randn(2, 4, 16).bfloat16().unbind(0)
y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
expected = (y * norm.weight.float() * torch.nn.functional.silu(gate.float())).bfloat16()
actual = norm(x, gate)
print(f'constructor norm={type(norm).__name__}; differing={(actual != expected).sum().item()}/{x.numel()}')
assert torch.equal(actual, expected)
