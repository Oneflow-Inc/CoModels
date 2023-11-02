import oneflow as flow
from utils.utils_callbacks import CallBackVerification
from backbones import get_model
from graph import TrainGraph, EvalGraph
import logging
import argparse
from utils.utils_config import get_config
from load_model import load_state_dict_from_url

def main(args):

    cfg = get_config(args.config)
    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)

    backbone = get_model(cfg.network, dropout=0.0, num_features=cfg.embedding_size).to(
        "cuda"
    )
    val_callback = CallBackVerification(
        1, 0, cfg.val_targets, cfg.ofrecord_path)

    if args.pretrained:
         if cfg.network == 'r50':
             url='http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/face_recognition/iresnet50.zip'
         elif cfg.network == 'r100':
             url='http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/face_recognition/iresnet101.zip'
         state_dict = load_state_dict_from_url(url)
    else:
         state_dict = flow.load(args.model_path)

    new_parameters = dict()
    for key, value in state_dict.items():
        if "num_batches_tracked" not in key:
            if key == "fc.weight":
                continue
            new_key = key.replace("backbone.", "")
            new_parameters[new_key] = value

    backbone.load_state_dict(new_parameters)

    infer_graph = EvalGraph(backbone, cfg)
    val_callback(1000, backbone, infer_graph)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace val")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    main(parser.parse_args())
