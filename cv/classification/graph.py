import oneflow as flow

def make_train_graph(
    model, cross_entropy, optimizer, lr_scheduler=None
):
    return TrainGraph(
        model, cross_entropy, optimizer, lr_scheduler
    )


def make_eval_graph(model):
    return EvalGraph(model)


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        cross_entropy,
        optimizer,
        lr_scheduler=None,
    ):
        super().__init__()

        self.config.allow_fuse_add_to_output(False)
        self.config.allow_fuse_model_update_ops(True)
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.world_size = flow.env.get_world_size()

        self.model = model
        self.cross_entropy = cross_entropy
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self, image, label):
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        loss = self.cross_entropy(logits, label)

        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, image):
        image = image.to("cuda")
        logits = self.model(image)
        return logits
