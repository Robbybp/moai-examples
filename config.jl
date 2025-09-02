import ArgParse

FILEDIR = dirname(@__FILE__)

function _add_common_args(settings::ArgParse.ArgParseSettings)
    ArgParse.add_arg_table(
        settings,
        "--index",
        Dict(
            :help=>"index of image (BASE-1!!!)",
            :default=>7, # TODO: Choose a reasonable default
            :arg_type=>Int,
        ),
        "--label",
        Dict(
            :help=>"Desired adversarial label",
            :default=>7, # TODO: Choose a reasonable default
            :arg_type=>Int,
        ),
        "--threshold",
        Dict(
            :help=>"Threshold for NN output of target label",
            :default=>0.6, # TODO: Choose a reasonable default
            :arg_type=>Float64,
        ),
        "--data-dir",
        Dict(
            :help=>"Directory to save data",
            :default=>joinpath(FILEDIR, "data"),
        ),
        "--solver",
        Dict(
            :help => "Solver for adversarial optimization problem $(VALID_OPTIMIZERS)",
            :default => "madnlp",
        ),
        "--reduced-space",
        Dict(
            :action => :store_true,
            :help => "Use a reduced-space (GrayBox) formulation",
        ),
    )
    return settings
end

function get_cli_settings()
    settings = ArgParse.ArgParseSettings()
    ArgParse.add_arg_table(settings, "fpath", Dict(:help=>"NN .pt file"))
    _add_common_args(settings)
    return settings
end

function get_profile_cli_settings()
    settings = get_cli_settings()
    _add_common_args(settings)
    ArgParse.add_arg_table(
        settings,
        "--nn",
        Dict(:help=>"ID of NN to use. Default loops over all NNs."),
    )
end

function _get_dir(name::String; create = true)
    dir = joinpath(FILEDIR, name)
    if isfile(dir)
        error("$dir is already a file")
    elseif !isdir(dir) && create
        mkdir(dir)
    end
    return dir
end

function get_nn_dir()
    return joinpath(FILEDIR, "nn-models")
end
