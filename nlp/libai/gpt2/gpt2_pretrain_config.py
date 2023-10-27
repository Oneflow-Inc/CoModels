from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .train import train
from .optim import optim
from .gpt_dataset import dataloader, tokenization
from .graph import graph
from omegaconf import DictConfig

from libai.models import GPTForPreTraining

#vocab_file = "./data_test/gpt_data/gpt2-vocab.json"
#merge_files = "./data_test/gpt_data/gpt2-merges.txt"
#data_prefix = "./data_test/gpt_data/loss_compara_content_sentence"
vocab_file = "/data/dataset/gpt2_data/gpt2-vocab.json"
merge_files = "/data/dataset/gpt2_data/gpt2-merges.txt"
data_prefix = "/data/dataset/gpt2_data/loss_compara_content_sentence"




tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix

cfg = dict(
    hidden_layers=6,
    vocab_size=30522,
    hidden_size=384,
    ffn_hidden_size=1536,
    num_attention_heads=12,
    max_seq_length=1024,
    embedding_dropout_prob=0,
    attention_dropout_prob=0,
    output_dropout_prob=0,
    layernorm_epsilon=1e-5,
    initializer_range=0.02,
    use_scaled_init_for_output_weights=True,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

cfg = DictConfig(cfg)

model = LazyCall(GPTForPreTraining)(cfg=cfg)

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.hidden_layers = 6
model.cfg.max_seq_length = 1024

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

train.train_micro_batch_size = 4
train.amp.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/gpt2_output"
