# 06_run_simulation.jl

agent_records = DataFrame(step=Int[], id=Int[], pos=Any[], toxicload=Float64[])

for step in 1:10
    Agents.step!(model, agent_step!, model_step!, 1)
    for agent in allagents(model)
        push!(agent_records, (step=step, id=agent.id, pos=agent.pos, toxicload=agent.toxicload))
    end
end

println(agent_records)

# Save to file
open("agent_records_output.txt", "w") do io
    write(io, "step\tid\tpos\ttoxicload\n")
    for row in eachrow(agent_records)
        pos_str = "($(row.pos[1]), $(row.pos[2]))"
        write(io, "$(row.step)\t$(row.id)\t$pos_str\t$(row.toxicload)\n")
    end
end
