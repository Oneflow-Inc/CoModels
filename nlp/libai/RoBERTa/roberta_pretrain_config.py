from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.models import RobertaForPreTraining
from omegaconf import DictConfig
from .graph import graph
from .train import train
from .optim import optim
from .roberta_dataset import dataloader, tokenization

vocab_file = "/data/dataset/roberta_data/roberta-vocab.json"
merge_files = "/data/dataset/roberta_data/roberta-merges.txt"
data_prefix = "/data/dataset/roberta_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix

cfg = dict(
    vocab_size=50265,
    hidden_size=768,
    hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=514,
    num_tokentypes=1,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    pad_token_id=1,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

cfg = DictConfig(cfg)



model = LazyCall(RobertaForPreTraining)(cfg=cfg)

# RoBERTa model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.input_placement_device = "cpu"

# parallel strategy settings
train.dist.data_parallel_size = 8
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_micro_batch_size = 2

train.amp.enabled = True

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_position_embeddings

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "output/roberta_output"
