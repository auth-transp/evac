using Agents, Agents.Pathfinding
using Random
using ColorTypes
import ImageMagick
using FileIO: load
using GLMakie 
using InteractiveDynamics
using Images
using DataFrames

@agent Agent ContinuousAgent{2} begin
    toxicload::Float64
end

v00 = 1.35 # m/s
toxicload = 0

if toxicload >0
    v00 = 1.35
elseif 0< toxicload <=1
    v00 = 1.35*exp(0.393*toxicload)
elseif 1< toxicload <3
    v00 = -1.78*ln(toxicload) + 2.063
elseif toxicload >=3
    v00 = 0 
end


