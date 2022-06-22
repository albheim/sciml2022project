cd(@__DIR__) # Cd to dir containing this file
import Pkg
Pkg.activate(".") # Activate local environment

using DelimitedFiles, Plots, Statistics
using Optimization, OptimizationFlux, OptimizationOptimJL, DiffEqSensitivity
using Symbolics, DataDrivenDiffEq, ModelingToolkit
using SymbolicRegression, Flux

include("CubicSplines.jl")
include("standard_models.jl")

# Read data
p_data = readdlm("data/pressure.csv", ',', Float64)[:, 2] # [mmHg]
ϕ_data = (60 / 1000) .* readdlm("data/flow.csv", ',', Float64)[:, 2] # L/min
const h = 0.005
const tv = 0:h:h*(length(p_data)-1)

# Spline interpolations, to get C2 input signal ϕc
αϕ = 1 - 1e-12 
αp = αϕ 
const ϕc = CubicSpline(collect(tv), ϕ_data, periodic=true, α=αϕ)
const pc = CubicSpline(collect(tv), p_data, periodic=true, α=αp)

# Define callback with plot and print
function callback(ps, l, pest)
    if isinteractive() 
        p = plot(tv, pest)
        scatter!(p, tv, pc.(tv))
        display(p)
    end
    @show l 
    false
end