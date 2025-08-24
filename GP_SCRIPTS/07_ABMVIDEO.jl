# 08_export_video.jl

using CairoMakie: Figure
import InteractiveDynamics: make_abm_video  # <-- Αυτό ΕΙΝΑΙ το ΚΛΕΙΔΙ!
CairoMakie.activate!()  # ΜΟΝΟ αν θες να εξαναγκάσεις CairoMakie ως backend

# Συνάρτηση χρώματος agent
function personcolor(agent::AgentEscapes)
    if agent.toxicload >= 3
        return :red
    elseif agent.toxicload <= 1
        return :green
    else
        return :orange
    end
end

agentcolor = personcolor  # Χρώμα agent
personcolor = personcolor  # Χρώμα agent για το abmplot!

# Ρυθμίσεις εμφάνισης agents
agent_attributes = Dict(
    :color => personcolor,
    :marker => :circle,
    :markersize => 9
)


function make_abm_video(agent_records::DataFrame,
                        heightmap::AbstractMatrix{<:Number};
                        filename::AbstractString="simulation.mp4",
                        framerate::Int=10)

    fig = Figure(resolution = (800, 600))
    ax  = Axis(fig[1, 1],
               xlimits = (1, size(heightmap, 2)),
               ylimits = (1, size(heightmap, 1)),
               aspect   = DataAspect())

    # Προαιρετικό background
    image!(ax, heightmap'; colormap = :gray)

    # Τα βήματα της προσομοίωσης
    steps = sort(unique(agent_records.step))

    # Κάνουμε το record
    record(fig, filename, steps;
           framerate     = framerate,
           video_encoder = FFMPEG.ffmpeg) do step

        clear!(ax)
        image!(ax, heightmap'; colormap = :gray)

        df = filter(r -> r.step == step, agent_records)

        xs   = getindex.(df.pos, 1)
        ys   = getindex.(df.pos, 2)
        cols = map(tl -> tl ≥ 3   ? :red
                        : tl ≤ 1 ? :green
                                 : :orange,
                   df.toxicload)

        scatter!(ax, xs, ys; color = cols, markersize = 12)
    end

    println("Έτοιμο το βίντεο: $filename")
end
