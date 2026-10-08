"""Batch independent crops without changing RandomResizedCrop's sampling."""

from collections import defaultdict

import torch
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F


def grouped_random_resized_crop(images, transform):
    """Sample each NCHW image independently, then resize equal-size crops together.

    Use the original transform's RNG, rejection sampling, fallback, interpolation
    and antialias setting. Restore image order after grouping. Custom transforms
    (including RandomResizedCrop subclasses) retain their original forward path.
    """
    if type(transform) is not RandomResizedCrop:
        return torch.stack([transform(image) for image in images], dim=0)

    groups = defaultdict(list)
    for index, image in enumerate(images):
        top, left, height, width = transform.get_params(image, transform.scale, transform.ratio)
        groups[height, width].append((index, top, left))

    output = images.new_empty((images.shape[0], images.shape[1], *transform.size))
    for (height, width), members in groups.items():
        crops = torch.stack([
            images[index, :, top:top + height, left:left + width]
            for index, top, left in members
        ], dim=0)
        resized = F.resize(crops, transform.size, transform.interpolation,
                           antialias=transform.antialias)
        indices = torch.tensor([member[0] for member in members],
                               dtype=torch.long, device=images.device)
        output.index_copy_(0, indices, resized)

    return output
