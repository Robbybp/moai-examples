import argparse
import torch
import torchvision
from torchvision.transforms.v2 import PILToTensor, ToTensor
import os

import config


torch.manual_seed(101)


ACTIVATION_LOOKUP = {
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "relu": torch.nn.ReLU,
    "softplus": torch.nn.Softplus,
    "softmax": torch.nn.Softmax,
}


def _get_fname(args):
    return f"mnist-{args.activation}{args.nodes}nodes{args.layers}layers.pt"


def predict(nn, x):
    output = nn(x)
    output_dim = len(output)
    return max(range(output_dim), key=lambda i: output[i])


def evaluate_accuracy(nn, dataset):
    nsamples = len(dataset)
    input_dim = torch.prod(torch.tensor(dataset[0][0].shape))
    x = dataset[0][0].reshape(input_dim)
    output_dim = len(nn(x))
    inputs = [image.reshape(input_dim) for image, _ in dataset]
    inputs = torch.stack(inputs)
    outputs = nn(inputs)
    predictions = torch.argmax(outputs, dim=1)
    labels = torch.tensor([label for _, label in dataset])
    ncorrect = torch.sum(labels == predictions)
    return float(ncorrect / nsamples)


def main(args):
    #transform = PILToTensor() # <- This doesn't scale the tensor
    transform = ToTensor()     # <- This scales the tensor
    train_dataset = torchvision.datasets.MNIST(
        # TODO: make dir CLI-configurable?
        root=config.get_data_dir(),
        train=True,
        transform=transform,
    )

    # "DataLoader combines a dataset and a sampler, and provides an iterable over
    # the given dataset"
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
    )

    _, image_height, image_width = train_dataset.data.shape
    input_dim = image_height * image_width
    hidden_dim = args.nodes
    n_hidden = args.layers
    output_dim = 10
    activation_function = ACTIVATION_LOOKUP[args.activation]

    # Implement architecture of NN:
    # - Affine layer mapping input dimension to hidden dimension
    # - (n-1) layers of: activation function followed by hidden-by-hidden affine layer
    # - activation layer
    # - hidden-by-output affine layer
    # - softmax
    layers = [torch.nn.Linear(input_dim, hidden_dim)]
    for i in range(n_hidden):
        layers.append(activation_function())
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
    layers.append(activation_function())
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    # We will apply softmax later, as it is easier to apply a "cross-entropy loss"
    # if we leave it out.
    # The outputs of this NN are "raw logits" (scores)
    nn = torch.nn.Sequential(*layers)

    # Example prediction
    if False:
        image, label = train_dataset[100]
        y = nn(image.reshape(input_dim))
        pred = max(range(output_dim), key=lambda i: y[i])
        print("Example prediction on image 100:")
        print(f"output = {y}")
        print(f"Prediction: {pred}")

    compute_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)
    nn.train() # Set model in training mode. Not sure why...
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            inshape = inputs.shape
            # inshape[0] is the batch size, potentially truncated because we're at the
            # end of the data
            inputs = inputs.reshape(inshape[0], input_dim)
            outputs = nn(inputs)
            #labels.to(float)
            #expected = torch.nn.functional.one_hot(labels)
            # I guess CrossEntropyLoss accepts labels directly?
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        ave_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} / {args.epochs}: Loss = {ave_loss:1.2e}")

    likelihood_predictor = torch.nn.Sequential(nn, torch.nn.Softmax(dim=0))
    acc = evaluate_accuracy(nn, train_dataset)
    ntrain = len(train_dataset)
    print(f"Accuracy on training set of {ntrain} samples: {acc}")

    # Evaluate on test data
    test_dataset = torchvision.datasets.MNIST(
        root=config.get_data_dir(),
        transform=transform,
        train=False,
    )
    nn.eval()
    acc = evaluate_accuracy(nn, test_dataset)
    ntest = len(test_dataset)
    print(f"Accuracy on test set of {ntest} samples: {acc}")

    fname = _get_fname(args)
    fpath = os.path.join(config.get_nn_dir(create=True), fname)
    if args.dry_run:
        print(f"--dry-run set. Not saving. Would have saved to {fpath}")
    else:
        print(f"Saving network to {fpath}")
        torch.save(likelihood_predictor, fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default=128, type=int, help="Nodes per layer. (default=128)")
    parser.add_argument("--layers", default=4, type=int, help="Number of layers. (default=4)")
    parser.add_argument("--activation", default="relu", help=f"Activation function. (default=relu. options={list(ACTIVATION_LOOKUP)})")
    parser.add_argument("--dry-run", action="store_true", help="Don't save trained network")
    parser.add_argument("--device", default="cpu", help="default='cpu'")
    parser.add_argument("--epochs", type=int, default=10, help="default=10")
    parser.add_argument("--batchsize", type=int, default=64, help="default=64")
    args = parser.parse_args()
    main(args)
