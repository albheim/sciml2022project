cd(@__DIR__) # Cd to dir containing this file
import Pkg
Pkg.activate(".") # Activate local environment

using DelimitedFiles, Plots, Statistics, ArgParse
using Optimization, OptimizationFlux, OptimizationOptimJL, DiffEqSensitivity
using Symbolics, DataDrivenDiffEq, ModelingToolkit
using SymbolicRegression, Flux

include("CubicSplines.jl")
include("standard_models.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--order"
        help = "What order the DE should be, default is 1."
        arg_type = Int
        default = 1
    "--repeats"
        help = "How many times to repeat the data, can help to enforce periodicity. Default is 1."
        arg_type = Int
        default = 1
end
parsed_args = parse_args(ARGS, s)

repeats = parsed_args["repeats"]
nstate = parsed_args["order"]

# Read data
p_data = repeat(readdlm("data/pressure.csv", ',', Float64)[:, 2], outer=repeats) # [mmHg]
ϕ_data = repeat((60 / 1000) .* readdlm("data/flow.csv", ',', Float64)[:, 2], outer=repeats) # L/min
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