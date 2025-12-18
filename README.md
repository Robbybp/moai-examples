MathOptAI examples
==================

This repository was intended to hold examples of using MathOptAI.jl.
Now it contains code to reproduce the results of my papers using
these examples.

## Training neural networks
Both papers rely on trained neural networks.
To train them, you will need a Python environment, for example:
```sh
python -m venv myenv
source myenv/bin/activate
pip install torch torchvision pandas
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
python train-scopf.py data/scopf/10_10_gen_load_input_v2.pt data/scopf/10_10_freq_output_v2.pt --nodes=500 --layers=5 --activation=tanh
```
The NNs will be saved in the `nn-models` directory.

The second paper relies on neural networks from the
[LoadShedVerification](https://github.com/SamChevalier/LoadShedVerification.git)
repository.
Clone this repository, then copy (or symlink) the `src/outputs/118_bus`
folder into the `nn-models/lsv` folder here. The relevant directory structure
should look like this:
```console
nn-models
├── lsv
│   └── 118_bus
│       ├── 118_bus_128node.pt
│       ├── 118_bus_2048node.pt
│       ├── 118_bus_32node.pt
│       ├── 118_bus_512node.pt
│       ├── 118_bus_normalization_values.h5
│       └── data_file_118bus.h5
...
```

## Julia environment
The Project.toml and Manifest.toml files used are included.
However, the manifest includes the local path of my `HSL_jll`,
which is important for reproducing these results.
I recommend downloading `HSL_jll` from the HSL website, `dev`-ing it locally,
then `resolve`-ing the manifest. E.g.:
```sh
julia --project=.
# Add or dev your local HSL_jll!
]resolve
```

Paper 1: "Nonlinear optimization with GPU-accelerated..."
---------------------------------------------------------

Reproducing the results of "Nonlinear optimization with GPU-accelerated
neural network constraints", presented at the ScaleOpt workshop at
NeurIPS 2025.

## Reproduce the optimization results

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

Paper 2. "Exploiting block-triangular submatrices..."
-----------------------------------------------------

Reproducing the results of "Exploiting block-triangular submatrices in
symmetric indefinite linear systems" (in preparation).

1. Generate a table of neural network structures:
```sh
julia --project=. isti2025/analyze-nns.jl
python isti2025/write-latex.py isti2025/tables/nns.csv
```

2. Generate a table of optimization problem structures:
```sh
julia --project=. isti2025/structure-sweep.jl
python isti2025/write-latex.py isti2025/tables/structure.csv
```

3. Generate a table showing the sizes of submatrices in the Schur complement
decomposition:
```sh
julia --project=. isti2025/matrix-structure.jl
python isti2025/write-latex.py isti2025/tables/matrix-structure.csv
```

Optionally, before we run the main numerical experiment, we can test
different linear solvers on different test problems to get a sense
for which solvers performs best.
```sh
julia --project=. isti2025/solver-sweep.jl
python isti2025/write-latex.py isti2025/tables/linear-solvers.csv
```
Note that the `solver-sweep.jl` script is hard-coded to skip
the large MNIST-MA97, SCOPF-MA97, and LSV-MA57 model-solver
combinations. We have determined by trial and error that these
combinations lead to segfaults or overflows, and will save you the
trouble of experiencing these failures yourself.

The results should indicate that MA57 is most effective on MNIST,
while MA86 is most effective on SCOPF and LSV.
These model-solver pairs have been hard-coded into the main
result generation script.

4. To generate the main runtime results:
```sh
# The script is called this because it uses the first 10 KKT matrices from MadNLP
julia --project=. isti2025/madnlp-runtime-sweep.jl
python isti2025/summarize-results.py isti2025/tables/runtime.csv
python isti2025/write-latex.py isti2025/tables/runtime-summary.csv
```
This generates results for 10 matrix factorizations per model-NN-solver combination,
then compresses them by averaging for each model-NN-solver.

5. We then profile MA57 and MA86 in terms of FLOPS and fill-in on the KKT matrices
and relevant Schur complement submatrices:
```sh
julia --project=. isti2025/matrix-sweep.jl
python isti2025/write-latex.py isti2025/tables/fill-in.csv
```

6. Finally, we generate a breakdown of solve times with our Schur complement
solver:
```sh
julia --project=. isti2025/runtime-breakdown.jl
python isti2025/summarize-results.py isti2025/breakdown.csv
python isti2025/write-latex.py isti2025/tables/breakdown-summary.csv
```
