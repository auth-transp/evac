# Κοινό helper: γράφει μία γραμμή στο ../RunHistory.xlsx μέσω scripts/append_run_history.py.
# include("../append_run_history.jl") από κάθε "SCENARIO X" φάκελο.

function append_run_history(; scenario::AbstractString, runner_script::AbstractString,
                              csv_file::AbstractString, mp4_file::AbstractString,
                              hist_mp4_file::AbstractString = "",
                              pathfinding_time::Real, run_timestamp::AbstractString)
    repo_root = normpath(joinpath(@__DIR__, ".."))
    python_exe = get(ENV, "RUNHISTORY_PYTHON",
        raw"C:\Users\gavin\AppData\Local\Programs\Python\Python311\python.exe")
    script = joinpath(repo_root, "scripts", "append_run_history.py")
    xlsx_path = joinpath(repo_root, "RunHistory.xlsx")

    cmd = `$python_exe $script --xlsx $xlsx_path --timestamp $run_timestamp
           --scenario $scenario --runner-script $runner_script --csv $csv_file
           --mp4 $mp4_file --hist-mp4 $hist_mp4_file --pathfinding-time $pathfinding_time`
    try
        run(cmd)
    catch e
        @warn "Αποτυχία εγγραφής στο RunHistory.xlsx — συμπλήρωσέ το χειροκίνητα." exception = e
    end
end
