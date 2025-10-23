isti:
	# Collect information about NNs (independent of any optimization model)
	julia --project=. isti2025/analyze-nns.jl
	python isti2025/write-latex.py isti2025/tables/nns.csv
	# Collect information about the structure of optimization problems
	julia --project=. isti2025/structure-sweep.jl
	python isti2025/write-latex.py isti2025/tables/structure.csv
	# Collect structural information about matrices
	julia --project=. isti2025/matrix-structure.jl
	python isti2025/write-latex.py isti2025/tables/matrix-structure.csv
	# Compare runtime on different solvers to decide which solver to use for the different methods
	julia --project=. isti2025/solver-sweep.jl
	python isti2025/write-latex.py isti2025/tables/linear-solvers.csv
	# Get overall runtimes with our method and the baseline
	julia --project=. isti2025/madnlp-runtime-sweep.jl
	# Write the runtime results to one big text file
	python isti2025/write-latex.py isti2025/tables/runtime.csv
	# Aggregate results by combining different samples
	python isti2025/summarize-results.py isti2025/tables/runtime.csv
	python isti2025/write-latex.py isti2025/tables/runtime-summary.csv
	# Get runtime breakdowns. This is a separate script as I can only collect
	# these breakdowns for my decomposition method.
	julia --project=. isti2025/runtime-breakdown.jl
	python isti2025/summarize-results.py isti2025/tables/breakdown.csv
	python isti2025/write-latex.py isti2025/tables/breakdown-summary.csv
	# Get FLOPS and NNZ required for factorization of each different matrix
	julia --project=. isti2025/matrix-sweep.jl
	python isti2025/write-latex.py isti2025/tables/fill-in.csv
