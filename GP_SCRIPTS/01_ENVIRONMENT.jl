# 01_environment.jl

begin
    using Agents
    using Agents.Pathfinding
    using Random
    using ColorTypes
    using ImageMagick
    using FileIO: load
    using InteractiveDynamics
    using Images
    using DataFrames
    using Statistics
    using CairoMakie
    #using Makie
end

begin
    heightmap_data = load("Maps/Qatargas Map.jpg")
    heightmap_data = permutedims(channelview(heightmap_data), [2,3,1])[:,:,1]
    heightmap = floor.(Int, convert.(Float64, heightmap_data) * 255)
end

begin
    concentration_data = load("Maps/concentrationmap new.jpg")
    concentration_data = permutedims(channelview(concentration_data), [2,3,1])[:,:,1]
    concentrationmap = floor.(Int, convert.(Float64, concentration_data .* 500))
end