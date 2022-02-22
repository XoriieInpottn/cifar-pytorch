#!/usr/bin/env python3


from docset import DocSet
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from torchstocks.utils.image import ImageNet


class Cifar10Dataset(Dataset):

    def __init__(self, path: str, train):
        super(Cifar10Dataset, self).__init__()
        self.docs = DocSet(path, 'r')
        self.aug = iaa.Sequential([
            # iaa.WithColorspace(
            #     to_colorspace='HSV',
            #     from_colorspace='RGB',
            #     children=iaa.SomeOf((0, None), [
            #         iaa.WithChannels(0, iaa.Multiply((0.985, 1.015))),
            #         iaa.WithChannels(1, iaa.Multiply((0.3, 1.7))),
            #         iaa.WithChannels(2, iaa.Multiply((0.6, 1.4)))
            #     ])
            # ),
            iaa.Fliplr(0.5),
            iaa.Pad(4, keep_size=False),
            iaa.CropToFixedSize(32, 32),
        ]) if train else iaa.Identity()

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        doc = self.docs[i]
        image = doc['feature']
        image = self.aug(image=image)
        image = ImageNet.encode_image(image)
        label = doc['label']
        return {
            'image': image,
            'label': label
        }
