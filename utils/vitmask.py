import numpy as np


class Mask:
    def __init__(self, mask_ratio):
        self.mask_ratio = mask_ratio

    def mask(self, num_patch):
        num_mask_patch = (np.round(num_patch*self.mask_ratio)).astype(np.int_)
        masked = np.zeros(num_mask_patch)
        unmasked = np.ones(num_patch-num_mask_patch)
        mask = np.hstack([masked, unmasked])
        np.random.shuffle(mask)
        return mask, num_mask_patch
