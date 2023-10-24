from .resmlp_imagenet import train, optim, model, dataloader, graph
import flowvision.transforms as transforms
from flowvision.transforms import InterpolationMode
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy


train.load_weight = "./output_resmlp/model_0000099"
