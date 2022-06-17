cd("project/heart")
import Pkg
Pkg.activate(".")
using DelimitedFiles, Plots, Optimization, OptimizationFlux, OptimizationOptimJL

p = readdlm("pressure.csv", ',', Float64)[:, 2] # [mmHg]
ϕ = (60 / 1000) .* readdlm("flow.csv", ',', Float64)[:, 2] # L/min
h = 0.005
t = 0:h:h*(length(p)-1)

# Spline interpolations, to get C2 input signal ϕc
include("CubicSplines.jl")
αϕ = 1 - 1e-12 # FIXME: should be set using xval
αp = αϕ # FIXME: should be set using xval
ϕc = CubicSpline(collect(t), ϕ, periodic=true, α=αϕ)
pc = CubicSpline(collect(t), p, periodic=true, α=αp)

plot(t, p, label="p")
plot!(t, ϕ, label="ϕ")

############### FIRST ORDER MODEL #############################3
function wk4p!(dx, x, p, t)
    dx .= p[1] * x .+ p[2] * ϕc(t)
end

function loss(ps, p)
    A, B, C, D, u0 = ps
    prob = ODEProblem(wk4p!, [u0], (0, t[end]), (A, B))
    sol = solve(prob, saveat=h)
    pest = C * sol[1, :] + D * ϕc.(sol.t)
    sum(abs2, pest .- pc.(sol.t)), pest
end

function callback(ps, l, pest)
    t = range(0, step=h, length=length(pest))
    p = plot(t, pest)
    scatter!(p, t, pc.(t))
    display(p)
    @show l 
    false
end

ps = rand(5)

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, ps)
optsol = solve(optprob, ADAM(0.1), maxiters=1000, callback=callback)


optpar = optsol.u

hidden = 16
nn = Chain(Dense(1, hidden, tanh), Dense(hidden, hidden, tanh), Dense(hidden, 1))
ps_nn, re = Flux.destructure(nn)


function wk4p_nn!(dx, x, p, t)
    A, B = p[1:2]
    dx .= A * x .+ B * ϕc(t) .+ re(p[3:end])(x)
end

function loss_nn(ps, p)
    A, B, C, D, u0 = ps[1:5]
    prob = ODEProblem(wk4p_nn!, [u0], (0, t[end]), [A; B; ps[6:end]])
    sol = solve(prob, saveat=h)
    pest = C * sol[1, :] + D * ϕc.(sol.t)
    sum(abs2, pest .- pc.(sol.t)), pest
end

ps = [optpar; ps_nn]

optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, ps)
optsol = solve(optprob, ADAM(0.01), maxiters=100, callback=callback)

optprob = OptimizationProblem(optf, optsol.u)
optsol = solve(optprob, BFGS(), callback=callback)



############### SECOND ORDER MODEL #############################3
function wk4p!(dx, x, p, t)
    A = reshape(p[1:4], 2, 2)
    B = reshape(p[5:6], 2, 1)
    dx .= A * x + B * ϕc(t)
end

function loss(ps, p)
    A, B, C, D, u0 = ps
    prob = ODEProblem(wk4p!, [u0], (0, t[end]), (A, B))
    sol = solve(prob, saveat=h)
    pest = C * sol[1, :] + D * ϕc.(sol.t)
    sum(abs2, pest .- pc.(sol.t)), pest
end

function callback(ps, l, pest)
    t = range(0, step=h, length=length(pest))
    p = plot(t, pest)
    scatter!(p, t, pc.(t))
    display(p)
    @show l 
    false
end