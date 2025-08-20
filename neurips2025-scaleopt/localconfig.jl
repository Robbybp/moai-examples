FILEDIR = dirname(@__FILE__)

function _get_dir(name::String; create = true)
    dir = joinpath(FILEDIR, name)
    if isfile(dir)
        error("$dir is already a file")
    elseif !isdir(dir) && create
        mkdir(dir)
    end
    return dir
end

function get_table_dir(; create = true)
    return _get_dir("tables")
end

function get_results_dir(; create = true)
    return _get_dir("results")
end
