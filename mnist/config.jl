import ArgParse

FILEDIR = dirname(@__FILE__)

function get_cli_settings()
    settings = ArgParse.ArgParseSettings()
    ArgParse.add_arg_table(
        settings,
        "fpath",
        Dict(:help=>"NN .pt file"),
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
