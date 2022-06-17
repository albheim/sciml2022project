using DelimitedFiles, Plots, Optimization, OptimizationFlux, OptimizationOptimJL, Statistics

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

# Second  order model
function wk4p!(x, p, t)
    A, B = p
    A * x + B * ϕc(t)
end

function loss(ps, p)
    A = reshape(ps[1:4], 2, 2)
    B = reshape(ps[5:6], 2, 1)
    C = reshape(ps[7:8], 1, 2)
    D = ps[9:9]
    u0 = ps[10:11]
    prob = ODEProblem(wk4p!, u0, (0, t[end]), (A, B))
    sol = solve(prob, saveat=h)
    pest = C * Array(sol) + D * ϕc.(sol.t)'
    mean(abs2, pest .- pc.(sol.t)), pest
end

function callback(ps, l, pest)
    t = range(0, step=h, length=length(pest))
    p = plot(t, pest')
    scatter!(p, t, pc.(t))
    display(p)
    @show l 
    false
end

ps = [-10., 0, 0, -10, 1, 1, 1, -1, 1, 1, 1] #.+ 0.01 * randn(11)

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, ps)
optsol = solve(optprob, ADAM(0.05), maxiters=1000, callback=callback)

optprob = OptimizationProblem(optf, optsol.u)
optsol = solve(optprob, BFGS(), callback=callback)
