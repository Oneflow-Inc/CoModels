from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .train import train
from .optim import optim
from .t5_dataset import dataloader, tokenization
from .graph import graph

from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import T5Model, T5ForPreTraining

cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=6,
    num_attention_heads=16,
    intermediate_size=1536,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    embedding_dropout_prob=0.1,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    amp_enabled=False,
)

cfg = DictConfig(cfg)

#t5_model = LazyCall(T5Model)(cfg=cfg)

model = LazyCall(T5ForPreTraining)(cfg=cfg)

vocab_file ="/data/dataset/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "/data/dataset/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix

# T5-large model config
model.cfg.num_attention_heads = 12
model.cfg.hidden_size = 384
model.cfg.hidden_layers = 6

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.train_micro_batch_size = 16
train.amp.enabled = True

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "./output/t5_output"
