from torchvision import datasets
import config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", default=None, help="Directory to store data")
    parser.add_argument("--absolute", action="store_true", help="Path is absolute, instead of relative to file dir")
    # TODO: Implement
    args = parser.parse_args()
    datadir = config.get_data_dir(create=True) if args.to is None else args.to
    datasets.MNIST(root=config.get_data_dir(create=True), download=True)
