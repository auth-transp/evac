begin                                   # This is a block of code that will be executed together
    using Agents, Agents.Pathfinding    # Load the Agents and Agents.Pathfinding modules
using Random                            # Load the Random module
using ColorTypes                        # Load the ColorTypes module
using ImageMagick                       # Load the ImageMagick module
using FileIO: load                      # Load the load function from the FileIO module
using GLMakie                           # Load the GLMakie module
using InteractiveDynamics               # Load the InteractiveDynamics module
using Images                            # Load the Images module
using DataFrames                        # Load the DataFrames module
using Statistics                        # Load the Statistics module
end                                     # End of the block of code

@agent AgentEscapes ContinuousAgent{2} begin    # Define the AgentEscapes agent type
    age::Float64                                # Age of the agent - Float64
    mass::Float64                               # Mass of the agent - Float64
    toxicload::Float64                          # Toxic load of the agent - Float64
    pathX::Vector{Float64}                      # X-coordinate of the agent's path - Vector{Float64}
    pathY::Vector{Float64}                      # Y-coordinate of the agent's path - Vector{Float64}
    TL1::Vector{Float64}                        # Toxic Load 1 - Vector{Float64}
    TL2::Vector{Float64}                        # Toxic Load 2 - Vector{Float64}
    TL3::Vector{Float64}                        # Toxic Load 3 - Vector{Float64}
end                                             # End of the AgentEscapes agent type


    heightmap_data = load("Maps/Qatargas Map.jpg")                                      # Load the heightmap data
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]           # Permute the dimensions of the heightmap data
    heightmap = floor.(Int, convert.(Float64, heightmap_data)*255)                      # Convert the heightmap data to a 2D array of integers
    concentration_data = load("Maps/concentrationmap new.jpg")                          # Load the concentration data 
    concentration_data = permutedims(channelview(concentration_data ), [2,3,1])[:,:,1]  # Permute the dimensions of the concentration data
    concentrationmap = floor.(Int, convert.(Float64, concentration_data .*500))         # Convert the concentration data to a 2D array of integers
    
    dt = 1.   ## discrete timestep each iteration of the model          # Define the dt variable as 1
    seed = 123  ## seed for random number generator                     # Define the seed variable as 123
    n_agents = 3                                                        # Define the n_agents variable as 3
    toxicity_rate = 0.07                                                # Define the toxicity_rate variable as 0.07
    age_range = (22,60)                                                 # Define the age_range variable as a tuple of 22 and 60
    speed_range = (4.0,7.0)                                             # Define the speed_range variable as a tuple of 4.0 and 7.0
    speed = 5.                                                          # Define the speed variable as 5 
    mass_range = (50,80)                                                # Define the mass_range variable as a tuple of 50 and 80
    ag_range_y = (size(heightmap)[1]/2-50):(size(heightmap)[1]/2+50)    # Define the ag_range_y variable as a range of values from the heightmap array # [1] stands for the 1st row
    ag_range_x = (size(heightmap)[2]/2-50):(size(heightmap)[2]/2+50)    # Define the ag_range_x variable as a range of values from the heightmap array # [2] stands for the 2nd row
    MW = 34 #Molecular weight of H2S in g/mol
    dims = (size(heightmap))                                            # Define the dims variable as the dimensions of the heightmap array (2xn matrix)
    walkmap = BitArray(trues(dims...))                                  # Define the walkmap variable as a BitArray of true values with the dimensions of the heightmap array
 

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

function setupToxic()                                               # Define the setupToxic function
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33] #min                # Define the Atime array as a 1x6 matrix                                    # Initialization of the 5 standard AEGL exposure times
    Arho = zeros(3, 6)                                              # Define the Arho array as a 3x6 matrix                                     # the concentration of the three symptoms compared to the AEGL concentrations
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]                   # Define the Arho array for the first row and columns 2 to 6                # odor
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]         # Define the Arho array for the second row and columns 2 to 6               # irritation
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71] #ppm    # Define the Arho array for the third row and columns 2 to 6                # edema
    MW = 34 #Molecular Weight of H2S
    Arho *= MW/24.04 #mg/m^3                                        # Multiply the Arho array by the molecular weight of H2S divided by 24.04
    Arho = Arho'                                                    # Transpose the Arho array 
    Atime = Atime*60 #seconds                                       # Multiply the Atime array by 60 seconds to convert to seconds              
    taumin = 200.                                                   # Define the taumin variable as 200 seconds (Borris&Patnaik, 2014)          # shortest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    taumax = 86400.                                                 # Define the taumax variable as 86400 seconds (Borris&Patnaik, 2014)        # longest exposure time over which an AEGL 1, 2 or 3 onset can be reached
    Brho = zeros(7,3)                                               # Define the Brho array as a 7x3 matrix of zeros                            # in ppm for each AEGL band at every time step ‘Atime’
    Balpha = zeros(7,3)                                             # Define the Balpha array as a 7x3 matrix of zeros                          # power-law exponents
    rhomax = zeros(1,3)                                             # Define the rhomax array as a 1x3 matrix of zeros                          # maximum concentration of H2S exposed by each agent
    rhomin = zeros(1,3)                                             # Define the rhomin array as a 1x3 matrix of zeros                          # minimum concentration of H2S exposed by each agent
    Btime = zeros(7, 3)                                             # Define the Btime array as a 7x3 matrix of zeros                           # represents an array, which is function of ‘taumin’ and ‘taumax’, that changes depending on alpha, which is a corresponsing array of power low exponents interpolating the ‘Brho’ table array

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
            for i = 2:eachindex(Btime)
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


InteractiveDynamics.abmvideo(
    "GP_TRIAL_1.mp4",
model,
    agent_step!,
    model_step!;
    figurekwargs = (resolution = size(heightmap),),
    frames = 400,
    framerate = 15,
    ac = personcolor,
    as = 8,
    figurekwargs = (resolution = size(model.properties[:heightmap]),),
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