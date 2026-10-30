# Διεργασία προσομοίωσης: τρέχει από τον Live Service σε ξεχωριστή διεργασία Julia (ώστε να ακυρώνεται με kill
# και να μη μπλοκάρει τον server).  Χρήση:  julia --project=webapp webapp/sim_worker.jl <φάκελος project> [T] [seed]
include(joinpath(@__DIR__, "src", "Simulation.jl"))
using .LiveSim

dir = ARGS[1]
T = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2500
seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 123

try
    println(LiveSim.run_simulation(dir; T = T, seed = seed))
catch e
    # η αποτυχία καταγράφεται στο progress.json ώστε να τη δει ο browser
    LiveSim.write_json(joinpath(dir, "progress.json"), (; status = "failed", frame = 0, T, error = sprint(showerror, e)))
    showerror(stderr, e, catch_backtrace())
    exit(1)
end
