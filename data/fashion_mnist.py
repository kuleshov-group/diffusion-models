from torchvision import transforms as tsfm
from torch.utils.data import DataLoader
from datasets import load_dataset

class FashionMNISTConfig:
    name = 'fashion_mnist'
    img_dim = 28 # dimensionality of the input image
    img_channels = 1 # number of image channels (here, greyscale)  
    unet_channels = 64 # size of the unet along the channel dimension
    unet_mults = (1, 2, 4,) # unet scaling factor for each layer
    a_dim = 2
    batch_size = 128
    learning_rate = 1e-3
    optimizer = 'adam'
    timesteps = 200
    epochs = 20

def get_fashion_mnist(batch_size, train=True, labels=False):
    # load dataset from the hub
    dataset = load_dataset("fashion_mnist")

    # define image transformations (e.g. using torchvision)
    transform = tsfm.Compose([
                tsfm.RandomHorizontalFlip(),
                tsfm.ToTensor(),
                tsfm.Lambda(lambda t: (t * 2) - 1)
    ])

    def transforms(examples):
       examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
       del examples["image"]
       return examples

    dataset = dataset.with_transform(transforms)

    if not labels:
        dataset = dataset.remove_columns("label")

    if train:
        dataset = dataset['train']
    else:
        dataset = dataset['test']

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader