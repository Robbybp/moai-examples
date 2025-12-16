import os
import torch
import argparse
import random


random.seed(789)


def count_parameters(nn):
    # Number of trainable parameters
    return sum(p.numel() for p in nn.parameters() if p.requires_grad)


class SimulationData(torch.utils.data.Dataset):

    def __init__(self, inputs, outputs):
        self.x = inputs
        self.y = outputs
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


ACTIVATION_LOOKUP = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "gelu": torch.nn.GELU,
}


def main(args):
    sim_inputs = torch.load(args.inputs).to(dtype=torch.float32)
    freq_output = torch.load(args.freq_output).to(dtype=torch.float32)
    # Unused for now
    #stab_output = torch.load(args.stab_output).to(dtype=torch.float32)
    input_dim = sim_inputs.shape[1]
    output_dim = freq_output.shape[1]
    print(f"Input dimension:   {input_dim}")
    print(f"Output dimension:  {output_dim}")
    print(f"Input data shape:  {sim_inputs.shape}")
    print(f"Output data shape: {freq_output.shape}")

    data = SimulationData(sim_inputs, freq_output)

    # NOTE: Train-test split not used for now
    #
    #sim_sample_indices = list(range(len(data)))
    #shuffled_indices = random.sample(sim_sample_indices, len(data))
    ## 20% of the data is used for testing
    #n_test = len(data) // 5
    #n_train = len(data) - n_test
    #test_indices = shuffled_indices[:n_test]
    #train_indices = shuffled_indices[n_test:]
    #train_dataset = torch.utils.data.Subset(data, train_indices)
    #test_dataset = torch.utils.data.Subset(data, test_indices)

    # Ignoring batches for now
    #n_batches = 1
    #batch_size_train = n_train // n_batches
    #batch_size_test = n_test // n_batches
    #train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset) #, batch_size=batch_size_train)
    #test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset)   #, batch_size=batch_size_test)

    # TODO: Graceful error if activation function isn't recognized?
    Activation = ACTIVATION_LOOKUP[args.activation]

    if args.small:
        nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 500),
            Activation(),
            #torch.nn.Tanh(),
            # NOTE: multi-layer neural networks appear very slow in JuMP. Uncomment this
            # and change surrounding layer dimensions to generate a good example for
            # profiling.
            #torch.nn.Linear(100, 100),
            #torch.nn.Tanh(),
            torch.nn.Linear(500, output_dim),
        )
    else:
        dimensions = [input_dim]
        dimensions.extend([args.nodes] * (args.layers - 2))
        dimensions.append(output_dim)
        layers = []
        for i, dim in enumerate(dimensions[:-1]):
            # Add a linear layer to convert to next layer's dimension
            if i != 0:
                # If this is not the input layer, add activation function
                #layers.append(torch.nn.Tanh())
                layers.append(Activation())
            layers.append(torch.nn.Linear(dim, dimensions[i+1]))

        nn = torch.nn.Sequential(*layers)
        #    torch.nn.Linear(input_dim, 500),
        #    torch.nn.Tanh(),
        #    torch.nn.Linear(500, 1000),
        #    torch.nn.Tanh(),
        #    torch.nn.Linear(1000, output_dim),
        #)

    nparam = count_parameters(nn)
    print(f"Training neural network with {nparam} parameters")

    optimizer = torch.optim.Adam(nn.parameters())
    loss_fcn = torch.nn.MSELoss()
    nn.train() # Set model in training mode. Don't think this matters for now...

    total_nodes = args.nodes * args.layers

    if args.small:
        printevery = 100
    else:
        #printevery = 10
        printevery = 1000
    n_epoch = args.epochs

    small_loss_count = 0
    for epoch in range(n_epoch):
        # Note that we are using all of the data for training here, i.e. no train-test
        # split
        loss = loss_fcn(nn(data.x), data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() <= 0.01:
            small_loss_count += 1
        if small_loss_count >= 1000:
            print("Loss has been < 0.01 for 1000 epochs. Terminating training")
            break

        if epoch % printevery == 0:
            print(f"Epoch {epoch}: Loss={loss.item()}")

    suffix = "-small" if args.small else ""
    prefix = f"{args.activation}" if args.activation != "tanh" else ""
    if args.nn_dir is None or args.small:
        prefix += "-"
        fname = f"{prefix}test-smooth-nn{suffix}.pt"
        print(f"Writing NN model to {fname}")
        torch.save(nn, fname)
    else:
        if not os.path.isdir(args.nn_dir):
            os.mkdir(args.nn_dir)
        fname = f"{prefix}{args.nodes}nodes{args.layers}layers.pt"
        fpath = os.path.join(args.nn_dir, fname)
        print(f"Writing NN model to {fpath}")
        torch.save(nn, fpath)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("inputs", help=".pt file containing input data used for simulations")
    argparser.add_argument(
        "freq_output", help=".pt file containing frequency output data from simulations"
    )
    # Unused for now
    #argparser.add_argument(
    #    "stab_output", help=".pt file containing stability output data from simulations"
    #)
    argparser.add_argument(
        "--nn-fname",
        default=None,
        help="Name of .pt file to which to write trained neural network",
    )
    argparser.add_argument("--nodes", default=100, type=int, help="Nodes per layer")
    argparser.add_argument("--layers", default=3, type=int, help="Layers, including input and output")
    argparser.add_argument("--small", action="store_true", help="Create a small neural network. This overrides --nodes and --layers")
    argparser.add_argument("--nn-dir", default=os.path.join("nn-models", "scopf"), help="Directory to store saved NN models")
    argparser.add_argument("--activation", default="tanh", help="'tanh' or 'relu', Default = tanh")
    argparser.add_argument("--epochs", type=int, default=10000, help="Default = 10000")
    args = argparser.parse_args()
    main(args)
