import argparse
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.v2 import PILToTensor

import config


def main(args):
    transform = PILToTensor()
    # TODO: make dir CLI-configurable?
    train_dataset = torchvision.datasets.MNIST(
        root=config.get_data_dir(),
        train=True,
        transform=transform,
    )
    # TODO: Before training, I will want to normalize these values. Do this with DataLoader?
    # Or I can use the ToTensor transform to normalize.

    image, label = train_dataset[50123]
    import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    # "images" are 3rd-order tensors for some reason...
    # Often the input to an image classification net is a set of "snapshots" of
    # the image. E.g. the input to AlexNet is 224x224x3. Maybe this is the motivation
    # here. Not sure.
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"Label = {label}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
