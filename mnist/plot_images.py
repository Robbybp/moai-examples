import argparse
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToTensor
import os

import config
import train as TRAIN
# TRAIN.predict and TRAIN.evaluate_accuracy may be useful


def _get_fname(args):
    assert args.fpath.endswith(".pt")
    nnfname = args.fpath[:-3]
    fname = nnfname + "-top{args.nimages}.pdf"
    return fname


def main(args):
    #transform = PILToTensor() # <- This doesn't scale the tensor
    transform = ToTensor()     # <- This scales the tensor
    # Evaluate on test data
    test_dataset = torchvision.datasets.MNIST(
        root=config.get_data_dir(),
        transform=transform,
        train=False,
    )

    # Load NN model
    nn = torch.load(args.fpath, weights_only=False)
    assert isinstance(nn[-1], torch.nn.Softmax)

    nn.eval()
    acc = TRAIN.evaluate_accuracy(nn, test_dataset)
    ntest = len(test_dataset)
    print(f"Accuracy on test set of {ntest} samples: {acc}")

    fname = _get_fname(args)
    fpath = os.path.join(config.get_nn_dir(create=True), fname)
    if args.dry_run:
        print(f"--dry-run set. Not saving. Would have saved to {fpath}")
    else:
        print(f"Saving figure to {fpath}")
        # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", help=".pt file containing NN")
    parser.add_argument("--nimages", default=16, help="Number of images to plot")
    args = parser.parse_args()
    main(args)
