#!/usr/bin/env python3


from docset import DocSet
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from torchcommon.utils.image import normalize_image, hwc_to_chw


class BaseDataset(Dataset):

    def __init__(self, path: str, aug: iaa.Augmenter):
        super(BaseDataset, self).__init__()
        self.docs = DocSet(path, 'r')
        self.aug = aug

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        doc = self.docs[i]
        image = doc['image']
        image = self.aug(image=image)
        image = normalize_image(image, transpose=True)
        label = doc['label']
        return {
            'image': image,
            'label': label
        }


class TrainDataset(BaseDataset):

    def __init__(self, path: str):
        super(TrainDataset, self).__init__(
            path=path,
            aug=iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Pad(4, keep_size=False),
                iaa.CropToFixedSize(32, 32),
            ])
        )


class TestDataset(BaseDataset):

    def __init__(self, path: str):
        super(TestDataset, self).__init__(
            path=path,
            aug=iaa.Identity()
        )
