import DataFrames
import CSV
if length(ARGS) < 1
    error("Provide a CSV file as the first argument")
elseif length(ARGS) > 1
    @warn "Additional arguments (beyond $(ARGS[1])) are ignored"
end
df = DataFrames.DataFrame(CSV.File(ARGS[1]))
println(df)
