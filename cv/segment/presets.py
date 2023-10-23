import oneflow as flow


def get_modules():

    import transforms

    return transforms, None, None


class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        backend="pil",
    ):
        T, tv_tensors, v2_extras = get_modules()

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size))]

        if hflip_prob > 0:
            transforms += [T.RandomHorizontalFlip(hflip_prob)]

        transforms += [T.RandomCrop(crop_size)]

        if backend == "pil":
            transforms += [T.PILToTensor()]

        transforms += [T.ToDtype(flow.float, scale=True)]

        transforms += [T.Normalize(mean=mean, std=std)]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil", use_v2=False
    ):
        T, _, _ = get_modules()

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.RandomResize(min_size=base_size, max_size=base_size)]

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [
            T.ToDtype(flow.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
        
        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
