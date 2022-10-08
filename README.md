# Cifar-10/100 Pytorch Implementation

## Dependency

```
torch
tqdm
numpy
imgaug
docset
sklearn
opencv-python
scikit-learn
torchstocks (*internal package)
```

## Dataset

The dataset is packed as DocSet files.

```
cifar-10/
	train.ds
	test.ds
cifar-100/
	train.ds
	test.ds
```

For each directory, there are two ds files:

```
train.ds
Count: 50000, Size: 151.9 MB, Avg: 3.1 KB/sample

Sample 0
    "filename": "bos_taurus_s_000507.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 19
    "coarse_label": 11
Sample 1
    "filename": "stegosaurus_s_000125.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 29
    "coarse_label": 15
...
Sample 49999
    "filename": "mako_s_001274.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 73
    "coarse_label": 1
```

```
test.ds
Count: 10000, Size: 30.4 MB, Avg: 3.1 KB/sample

Sample 0
    "filename": "volcano_s_000012.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 49
    "coarse_label": 10
Sample 1
    "filename": "woods_s_000412.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 33
    "coarse_label": 10
...
Sample 9999
    "filename": "rose_s_000753.png"
    "image": ndarray(dtype=uint8, shape=(32, 32, 3))
    "label": 70
    "coarse_label": 2
```

## Some Results

| Dataset   | Backbone  | NonLin. | Acc.   | Optimizer | Learning rate   | Weight decay | Batch size | Epochs |
| --------- | --------- | ------- | ------ | --------- | --------------- | ------------ | ---------- | ------ |
| Cifar-100 | vgg16     | ReLU    | 72.69% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | vgg19     | ReLU    | 70.79% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet18  | ReLU    | 76.95% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet34  | ReLU    | 78.45% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet50  | ReLU    | 79.02% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet101 | ReLU    | 78.44% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | vgg16     | SiLU    | 73.34% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | vgg19     | SiLU    | 70.59% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet18  | SiLU    | 77.44% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |
| Cifar-100 | resnet34  | SiLU    | 78.52% | AdamW     | 1e-3->0, Cosine | 0.3          | 32         | 100    |

