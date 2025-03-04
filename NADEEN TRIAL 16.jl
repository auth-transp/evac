begin
    using Agents, Agents.Pathfinding
using Random
using ColorTypes
using ImageMagick
using FileIO: load
using GLMakie 
using InteractiveDynamics
using Images
using DataFrames
using Statistics
end

@agent AgentEscapes ContinuousAgent{2} begin
    age::Float64
    mass::Float64
    toxicload::Float64
    pathX::Vector{Float64}
    pathY::Vector{Float64}
    TL1::Vector{Float64}
    TL2::Vector{Float64}
    TL3::Vector{Float64}
end


    heightmap_data = load("Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)
    concentration_data = load("Maps/concentrationmap new.jpg")
    concentration_data = permutedims(channelview(concentration_data ), [2,3,1])[:,:,1]
    concentrationmap = floor.(Int, convert.(Float64, concentration_data .*500))
    
    dt = 1.   ## discrete timestep each iteration of the model
    seed = 123  ## seed for random number generator
    n_agents = 3
    toxicity_rate = 0.07
    age_range = (22,60)
    speed_range = (4.0,7.0)
    speed = 5.
    mass_range = (50,80)
    ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)
    ag_range_x = (size(heightmap)[2]/2-50):(size(heightmap)[2]/2+50)
    MW = 34 #Molecular weight of H2S in g/mol
    dims = (size(heightmap))
    walkmap = BitArray(trues(dims...))
 

    #goals
    dests = [(600., 980.), (100., 200.)]

    #Generate the RNG for the model
    rng = MersenneTwister(seed)

    ## Note that the dimensions of the space do not have to correspond to the dimensions
    ## of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace(size(heightmap); periodic = false, spacing = 1)
    

    pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}()))
    properties = (
        pathfinder = AStar(space; walkmap = walkmap, cost_metric = PenaltyMap(heightmap, MaxDistance{2}())),
        heightmap = heightmap, 
        dt = dt,
        speed_range = speed_range,
        :goal => dests,
    )

   
    model = ABM(AgentEscapes, space; rng, properties)
    

    for _ in 1:n_agents
        age = rand(abmrng(model))*(age_range[2]-age_range[1])
        mass = rand(abmrng(model)) * (mass_range[2]-mass_range[1]) +mass_range[1]
        vel = Tuple(rand(abmrng(model), 2) .* (speed_range[2]-speed_range[1]) .+ speed_range[1])
        pos = Tuple((rand(abmrng(model), floor.(ag_range_y)), rand(abmrng(model), floor.(ag_range_x))))
        person = add_agent!(pos, AgentEscapes, model, vel, age, mass, 1., [pos[1]], [pos[2]], [0.0], [0.0], [0.0])
        plan_best_route!(person, dests, model.pathfinder)
    end
                
    return model
   


function calculate_dispersion(heightmap)
    #dims = (size(heightmap))
    #return rand!(model.rng,zeros(dims))
    return concentrationmap
end
concentrationmap = calculate_dispersion(heightmap)  # Calculate toxic load based on position

function setupToxic()
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33] #min 
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71] #ppm
    MW = 34 #Molecular Weight of H2S
    Arho *= MW/24.04 #mg/m^3
    Arho = Arho'
    Atime = Atime*60 #seconds
    taumin = 200.
    taumax = 86400.
    Brho = zeros(7,3)
    Balpha = zeros(7,3)
    rhomax = zeros(1,3)
    rhomin = zeros(1,3)
    Btime = zeros(7, 3)

    #Initialize
    for k=1:3
        for b=2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin;
         Btime[7, k] = taumax;
    end    
    
    #power low trial values
    for k=1:3
        for b=3:6
            if Brho[b-1, k]==Brho[b, k]
                Balpha[b, k] = 0.0;
            else
                Balpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end

        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
        #Balpha[6, k] = Balpha[5, k]
    end
    #extrapolate for the edges
    for k=1:3
        if Balpha[3, k]==0
            rhomax[k] = Brho[2, k];
            Brho[1, k] = rhomax[k];
        else
            rhomax[k] = Brho[2, k]*(Btime[2, k]/taumin)^(1/Balpha[2, k]);
            Brho[1, k] = rhomax[k];
        end

        if Brho[5, k]==Brho[6, k]
            rhomin[k] = Brho[6, k];
            Brho[7, k] = rhomin[k];
        else
            rhomin[k] = Brho[6, k]*(Btime[6, k]/taumax)^(1/Balpha[6, k]); #I fixed Brho[6] to be Brho[6, k]
            Brho[7, k] = rhomin[k];
        end

    end
    #Correct to account for threshold values
    for k=1:3
        for b=2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k];
            end
        end
    end

    for k=1:3
        for b=3:5
            if Balpha[b-1, k]==0 && Balpha[b, k]>0
                Balpha[b, k]=log(Btime[b, k]/Btime[b-1, k])/log(Brho[b-1, k]/Brho[b, k]);
            end
        end
    end

    return Balpha, Btime, Brho'
end

Balpha, Btime, Brho = setupToxic()

function update_toxic_load(Ct, TLcurrent, dt)
   
    #a = alpha
    #Btime = Btime;
    #Brho = Brho;
    TL = TLcurrent
    TL_rate = 0.0

    for k = 1:3
        Cmin = Brho[k, 7]
        Cmax = Brho[k, 1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1];
        elseif Ct < Cmin
            TL_rate = 0.0;
        else
            for i = 2:length(Btime)
                if i-1 > 0 && i <= size(Brho, 2)
                    if Brho[k, i-1] < Ct < Brho[k, i] #check 
                        TL_rate = (1/Btime[i])*((Ct/Brho[k,i])^(Balpha[i-1])) #Toxic Load rate in units of s^-1 
                    end
                end
            end
        end
        TL[k] = TL[k] .+ TL_rate * dt;
        
    end
        #TL[iAg, :] .= sum(TL .> 1) .+ TL[min(3, sum(TL .> 1) + 1)] .* (1 - (TL[3] > 1));
        #TL[iAg, :] .= person.toxicload;
    TL[1] = TL[1] > 1.0 ? 1.0 : TL[1]
    TL[2] = TL[2] > 1. ? 1.0 : TL[2]
    TL[3] = TL[3] > 1. ? 1.0 : TL[3]

    return TL
end
 

# function update_toxic_load!(person::AgentEscapes, dt::Float64, toxicity_rate::Float64, heightmap::Array{Int, 2})
#     position = floor.(Int, person.pos)  # Get agent's position and convert to integer
#     # Update toxic load of the agent based on the three bands approach by Nawayd
#     person.toxicload += toxicity_rate * concentrationmap[position[1], position[2]] * dt
# end

function agent_step!(person, model)
    
    position = floor.(Int, person.pos)
    Ct = concentrationmap[position[1], position[2]]
    TLcurrent = [person.TL1[end], person.TL2[end], person.TL3[end]]
    TL = update_toxic_load(Ct, TLcurrent, dt)
    
    person.toxicload=sum(TL)
    push!(person.TL1, TL[1])
    push!(person.TL2, TL[2])
    push!(person.TL3, TL[3])

    speed = 1.35 #m/s
    if 0< person.toxicload <=1
         speed = 1.35*exp(0.393*person.toxicload)
    elseif 1< person.toxicload <3
        speed = -1.78*log(person.toxicload) + 2.063
    elseif person.toxicload >=3
         speed = 0. 
    end
    display("$speed - $(person.toxicload)")
    move_along_route!(person, model, model.pathfinder, speed, dt)
    push!(person.pathX, person.pos[1])
    push!(person.pathY, person.pos[2])
end


function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end



function static_preplot!(ac, model) 
    scatter!(ac, model.goal; color = (:red, 50), marker = 'o')
end


function personcolor(person)
    person.toxicload >= 3. ? :red : (person.toxicload <=1. ? :green : :orange)
end


abmvideo(
    "NADEEN TRIAL 16-H2S-Qatargas_.mp4", 
    model, 
    agent_step!,
    model_step!;
    figurekwargs = (resolution = size(heightmap),),
    frames = 400,
    framerate = 15,
    ac = personcolor,
    as = 8,
    scatterkwargs = (strokecolor = :white, strokewidth = 1),
    heatarray = model -> penaltymap(model.pathfinder),
    heatkwargs = (colormap = :grays,),
    static_preplot!
) 


#model = ABM(AgentEscapes, space; rng, properties)
#space2 = ContinuousSpace((915, 983); periodic = false, spacing = 1.0)
#model2 = StandardABM(AgentEscapes, space2, agent_step! = agent_step!, model_step! = model_step!, properties = properties, rng = MersenneTwister(seed))

# model2 = Agents.ABM(
#     AgentEscapes, space;
#     properties = properties,
#     agent_step! = agent_step!,
#     model_step! = model_step!
# )




function final_result(model)
    agent_step!;
    model_step!;
    model = ABM(AgentEscapes, space; rng, properties)
end

final_result(model) = model2

adata = [(:model2, f), :toxicload, :pos, :pathX, :pathY, :TL1, :TL2, :TL3]
adf, mdf = run!(AgentEscapes, 100., adata)

#model2 = run!(model, 100., adata = adata)
# adf, mdf = run!(model2, 100; adata = adata)

#Tests
# f = Figure()
# ax = GLMakie.Axis(f[1, 1], yscale = log10, xscale = log10) 
# x = Btime
# y = Brho
# scatter!(ax, x, y[1,:])
# scatter!(ax, x, y[2,:])
# scatter!(ax, x, y[3,:])
# f
# for (p,v) in model.agents
#     display(v.TL1')
# end