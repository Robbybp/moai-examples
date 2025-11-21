MathOptAI examples
==================

This repository was intended to hold examples of using MathOptAI.jl.
Now it contains code to reproduce the results of my papers using
these examples.

Paper 1: "Nonlinear optimization with GPU-accelerated..."
---------------------------------------------------------

Reproducing the results of "Nonlinear optimization with GPU-accelerated
neural network constraints", presented at the ScaleOpt workshop at
NeurIPS 2025.

# Training neural networks

First you will need a Python environment. For example:
```sh
python -m venv myenv
source myenv/bin/activate
pip install torch torchvision
```

Download data needed for the MNIST test problem:
```sh
python download_mnist.py
```
The small dataset used to train the SCOPF test problem is contained
in this repository in `data/scopf`.

Train neural networks, e.g.:
```sh
python train-mnist.py --nodes=1024 --layers=4 --activation=tanh --device=cpu
python train-scopf.py data/scopf/10_10_gen_load_input_v2.pt data/scopf/10_10_freq_output_v2.pt --nodex=500 --layers=5 --activation=tanh
```
The NNs will be saved in the `nn-models` directory.

# Reproduce the optimization results

You will need a Julia environment. The Project.toml and Manifest.toml files used
are included. However, the manifest includes the local path of my `HSL_jll`,
which is important for reproducing these results.
I recommend downloading `HSL_jll` from the HSL website, `dev`-ing it locally,
then `resolve`-ing the manifest. E.g.:
```sh
julia --project=.
# Add or dev your local HSL_jll!
]resolve
```

You can then reproduce the results:
1. Generate a table of NN structures:
```sh
julia --project=. neurips2025-scaleopt/analyze-nns.jl
```
2. Generate a table showing the structure (numbers of variables/constraints/nonzeros)
of each optimization problem:
```sh
julia --project=. neurips2025-scaleopt/analyze-structure.jl
```
3. Generate a table of runtime results:
```sh
julia --project=. neurips2025-scaleopt/analyze-runtime.jl
```
