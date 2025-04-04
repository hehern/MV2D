import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    # GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


class CustomGridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio_range=(0.6, 0.95), mode=0, prob=1.,
                 interv_ratio=0.5, masked_value=0):
        super(CustomGridMask, self).__init__()
        self.use_h = use_h#True
        self.use_w = use_w#True
        self.rotate = rotate#1
        self.offset = offset#False
        self.ratio_range = ratio_range#(0.4, 0.6)
        self.mode = mode#1
        self.st_prob = prob#0.7
        self.prob = prob
        self.interv_ratio = interv_ratio#0.8
        self.masked_value = masked_value#0

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        masks = []
        for i in range(n):
            d = np.random.randint(2, max(int(h * self.interv_ratio), 3))
            ratio = np.random.uniform(self.ratio_range[0], self.ratio_range[1])
            self.l = min(max(int(d * ratio + 0.5), 1), d - 1)
            mask = np.ones((hh, ww), np.float32)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)
            if self.use_h:
                for i in range(hh // d):
                    s = d * i + st_h
                    t = min(s + self.l, hh)
                    mask[s:t, :] *= 0
            if self.use_w:
                for i in range(ww // d):
                    s = d * i + st_w
                    t = min(s + self.l, ww)
                    mask[:, s:t] *= 0

            r = np.random.randint(self.rotate)
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.rotate(r)
            mask = np.asarray(mask)
            mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

            mask = torch.from_numpy(mask).float().cuda()
            # mode 0: the grid is masked out
            # mode 1: the
            if self.mode == 1:
                mask = 1 - mask
            # mask = mask.expand_as(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # [n, h, w]
        masks = masks.repeat_interleave(c, dim=0)  # [n*c, h, w]

        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * masks + offset * (1 - masks)
        else:
            x = x * masks
            if self.masked_value != 0:
                x = x + (1 - masks) * self.masked_value

        return x.view(n, c, h, w)
