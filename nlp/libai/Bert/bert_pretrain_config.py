from omegaconf import DictConfig
from libai.models import BertForPreTraining
from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from ..graph import graph
from ..train import train
from ..optim import optim
from .bert_dataset import dataloader, tokenization

vocab_file = "/data/dataset/bert_data/bert-base-chinese-vocab.txt"
data_prefix = "/data/dataset/bert_data/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
dataloader.test[0].dataset.data_prefix = data_prefix
dataloader.test[0].dataset.indexed_dataset.data_prefix = data_prefix



cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=24,
    num_attention_heads=12,
    intermediate_size=4096,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    num_tokentypes=2,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    add_binary_head=True,
    amp_enabled=False,
)

cfg = DictConfig(cfg)
model = LazyCall(BertForPreTraining)(cfg=cfg)

model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 768
model.cfg.hidden_layers = 8

train.input_placement_device = "cpu"

train.dist.pipeline_num_layers = model.cfg.hidden_layers

train.train_micro_batch_size = 16

train.amp.enabled = True

for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_position_embeddings

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.output_dir = "output/bert_output"
