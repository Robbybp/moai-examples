# MathOptAI examples
This has turned into a repository for the development of a MadNLP linear solver
that exploits the block-triangular structure implied by a neural network, represented
as a MathOptAI `PipelineFormulation`. We do this with the following steps:
1. A Schur complement decomposition that pivots on the Jacobian of the neural network
   constraints with respect to output and intermediate variables (and the transpose
   of this Jacobian)
2. A block-triangular decomposition on the pivot matrix. We use the block-triangular
   form defined by the neural network's layers. This decomposition requires that
   we do not regularize the pivot matrix's "dual block".
3. A block-diagonal decomposition of the diagonal blocks of the block-triangular
   decomposition. This is only necessary because Julia-UMFPACK appears to not
   properly exploit block-diagonal form.

> [!NOTE]
> While the decomposition works well in isolation, I see a factor of 5+ slowdown
> when using the linear solver in MadNLP. This used to be worse, but I improved
> it by removing many (but not all) allocations from my factorization/solve methods.
> The bottleneck now appears to be type inference.

Here is internal timing data collected (using `Base.time()`) when running
10 solves in isolation (via the `profile-linalg.jl` script):
```console
SchurComplementSolver timing information
----------------------------------------
initialize: 5.588640928268433
factorize:  30.58126950263977
  reduced:        0.08894991874694824
  pivot:          2.5815818309783936
  update_pivot:   0.2777388095855713
  solve:          14.7985200881958
  multiply:       12.223405122756958
  update_reduced: 0.008051633834838867
  other:          0.6030220985412598
solve:      6.536936044692993
----------------------------------------
```
And here it is when running 10 iterations of MadNLP (via the `test-nlp.jl` script):
```console
SchurComplementSolver timing information
----------------------------------------
initialize: 5.788884878158569
factorize:  172.30124473571777
  reduced:        30.989306211471558
  pivot:          1.313755989074707
  update_pivot:   0.0790853500366211
  solve:          85.70144510269165
  multiply:       48.567078828811646
  update_reduced: 1.2385411262512207
  other:          4.412032127380371
solve:      2.0836191177368164
----------------------------------------
```

## Reproducing these results

These results are produced with Julia 1.11.6 on MacOS 15 with an M1 Max and 32 GB RAM.
They rely on the Julia packages defined in Project and Manifest, including a local
copy of `HSL_jll`. They also rely on `PythonCall` using Python with torch installed.
Using `CondaPkg`, this can be done with (I think, it's been a while):
```
using CondaPkg
]conda add pytorch
```
The benchmarks (but not most tests) also rely on some NNs saved to disk as `.pt` files.
These are generated offline with the `train.py` Python script.

### Training neural networks

To train the NNs, we need a Python environment with `torch` and `torchvision`.
We could probably do this with the `CondaPkg` environment, but I just create a new
environment:
```
python3.12 -m venv myenv
source myenv/bin/activate
pip install torch torchvision
```
Then we download the MNIST dataset:
```
python download_mnist.py
```
This downloads the dataset to the `data` subdirectory. The Julia code relies
on it being located here.

Then we train the NNs:
```
python train.py --nodes=128 --layers=4 --activation=relu
python train.py --nodes=512 --layers=4 --activation=relu
python train.py --nodes=1024 --layers=4 --activation=relu
python train.py --nodes=2048 --layers=4 --activation=relu
# etc. I think the scripts currently just use 1024 and 2048.
```
This stores the trained NNs in the `nn-models` subdirectory. The Julia code
relies on them being located here.

### Running the Julia code

Ideally, we replicate my original environment using the manifest, but this contains
a hard-coded local path to my `HSL_jll`. Maybe this works:
```
julia --project=.
# add or dev your own HSL_jll somehow!
]resolve
```
If that worked, we should be able to run the code:
```
julia --project=. test-blockdiagonal.jl
julia --project=. test-btsolver.jl # This benchmarks on the 2048-by-4 NN, so it will take some time
julia --project=. test-linalg.jl
```
Then we can run the benchmarks:
```
julia --project=. profile-linalg.jl
julia --project=. test-nlp.jl
```

## Profiling

My main tools for debugging performance have been profiling runtime and
profiling allocations. To do this, I manually set either the `PROFILE_ALLOCS` or
`PROFILE_RUNTIME` flag in `test-nlp.jl`, e.g.:
```diff
diff --git a/test-nlp.jl b/test-nlp.jl
index daca6f3..9c354d4 100644
--- a/test-nlp.jl
+++ b/test-nlp.jl
@@ -101,7 +101,7 @@ madnlp_schur = JuMP.optimizer_with_attributes(
 )
 JuMP.set_optimizer(m, madnlp_schur)
 
-PROFILE_ALLOCS = false
+PROFILE_ALLOCS = true
 if PROFILE_ALLOCS
     Profile.Allocs.@profile sample_rate=0.0001 JuMP.optimize!(m)
     #Profile.print()
```
This writes profile data, which I can then view with `view-profile.jl`
(which may need another hard-coded flag to be set).
Runtime appears to be dominated by matrix multiplication, while allocations are
dominated by type inference (I think?).

> [!NOTE]
> The `types` branch contains my attempts at improving type stability.

I have recently tried using `InteractiveUtils.@code_warntype` to track down type instabilities.
See the `warntype.jl` script in the `types` branch.
The first type instabilities I noticed were due to `SchurComplementSolver`'s `pivot_solver`
and `reduced_solver` fields being subtypes of `AbstractLinearSolver`. I'm not sure if these
are likely to cause slowdowns, or what I can do about them.
