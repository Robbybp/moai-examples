import CSV
import DataFrames
using Printf

include("localconfig.jl")
include("setup-compare-formulations.jl")

results_dir = get_results_dir()

dfs = []
for i in 1:n_elements
    local fname = "runtime-$(@sprintf("%02d", i)).csv"
    #local fname = "runtime-$(@sprintf("%d", i)).csv"
    local fpath = joinpath(results_dir, fname)
    if isfile(fpath)
        local df = DataFrames.DataFrame(CSV.File(fpath))
        push!(dfs, df)
    else
        @warn "$fpath does not exist"
    end
end
df = reduce(vcat, dfs)
tabledir = get_table_dir()
fname = "runtime.csv"
fpath = joinpath(tabledir, fname)
println("Writing combined results to $fpath")
CSV.write(fpath, df)
println(df)
