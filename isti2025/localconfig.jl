include("../config.jl")
function get_table_dir()
    # ... _get_dir uses its own directory... this is not really what I want.
    # This isn't horrible, but it feels like it will get confusing.
    # If I define FILEDIR in this function, it's unclear which takes precedence...
    return _get_dir(joinpath("isti2025", "tables"))
end
