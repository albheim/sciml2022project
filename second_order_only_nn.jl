cd(@__DIR__) # Cd to dir containing this file
import Pkg
Pkg.activate(".") # Activate local environment

using DelimitedFiles, Plots, Statistics
using Flux, DiffEqSensitivity
using Optimization, OptimizationFlux, OptimizationOptimJL
using Symbolics, SymbolicRegression, DataDrivenDiffEq, ModelingToolkit

# Read data
p = readdlm("pressure.csv", ',', Float64)[:, 2] # [mmHg]
ϕ = (60 / 1000) .* readdlm("flow.csv", ',', Float64)[:, 2] # L/min
const h = 0.005
const t = 0:h:h*(length(p)-1)

# Spline interpolations, to get C2 input signal ϕc
include("CubicSplines.jl")
αϕ = 1 - 1e-12 
αp = αϕ 
const ϕc = CubicSpline(collect(t), ϕ, periodic=true, α=αϕ)
const pc = CubicSpline(collect(t), p, periodic=true, α=αp)

plot(t, p, label="p")
plot!(t, ϕ, label="ϕ")

# Define callback with plot and print
function callback(ps, l, pest)
    if isinteractive() 
        p = plot(t, pest)
        scatter!(p, t, pc.(t))
        display(p)
    end
    @show l 
    false
end

# Neural net 
hidden = 16
nnps, re = Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, tanh),
        Dense(hidden, hidden, tanh),
        Dense(hidden, nstate)
    )
)

function wk4p_nn(x, p, t)
    ϕ = ϕc(t)
    re(p)([x; ϕ])
end

function simulate_nn(ps, p)
    psall = [vec(A); vec(B); ps]
    prob = ODEProblem(wk4p_nn, u0, (0, t[end]), psall)
    solve(prob, saveat=h)
end

function loss_nn(ps, p)
    A, B, C, D, u0 = p
    sol = simulate_nn(ps, p)
    pest = vec(C * Array(sol) + D * ϕc.(sol.t)')
    mean(abs2, pest .- pc.(sol.t)), pest
end

ps = nnps
p = (A, B, C, D, u0)

# Is this faster with Zygote?
optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

optprob = OptimizationProblem(optf, ps, p)
optsol = solve(optprob, ADAM(0.05), maxiters=100, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, BFGS(), callback=callback)

# Do symbolic regression
sol = simulate_nn(optsol.u, p)
x = [Array(sol); ϕc.(sol.t)']
y = re(optsol.u[(length(optpar)+1):end])(x)

prob = DirectDataDrivenProblem(x, y)

@variables u[1:3]
basis = Basis([
    polynomial_basis(u, 3);
    #exp.(polynomial_basis(u, 2));
    sin.(polynomial_basis(u, 2));
], u)
println(basis)

res = solve(prob, basis, STLSQ(1.0))
println(res)
println(result(res))

# Check solution
eq = substitute(equations(result(res))[1].rhs, parameter_map(res))
generate_function(eq, u; expression=Val(true))

f = generate_function(result(res), states(result(res)), parameters(result(res)))

function wk4p_extended!(dx, x, p, t)
    wk4p!(dx, x, p, t) # Update according to base eq
    dx .+= f([x; ϕc(t)])
end


# Plotting
ns = nstate
ps = optpar
A = reshape(ps[1:ns^2], ns, ns)
B = reshape(ps[(ns^2+1):(ns^2+ns)], ns, 1)
C = reshape(ps[(ns^2+ns+1):(ns^2+2ns)], 1, ns)
D = ps[(ns+1)^2]
u0 = reshape(ps[((ns+1)^2+1):((ns+1)^2+ns)], ns)

p1 = plot(t, pc.(t), label="real")

prob_default = ODEProblem(wk4p, [u0], (0, t[end]), (A, B))
sol_default = solve(prob_default, saveat=h)
pest_default = C * sol_default[1, :] + D * ϕc.(sol_default.t)
plot!(p1, t, pest_default, label="default")

prob_nn = ODEProblem(wk4p_nn, [u0], (0, t[end]), (A, B, ps[6:end]))
sol_nn = solve(prob_nn, saveat=h)
pest_nn = C * sol_nn[1, :] + D * ϕc.(sol_nn.t)
plot!(p1, t, pest_nn, label="nn")

prob_extended = ODEProblem(wk4p_extended!, [u0], (0, t[end]), (A, B))
sol_extended = solve(prob_extended, saveat=h)
pest_extended = C * sol_extended[1, :] + D * ϕc.(sol_extended.t)
plot!(p1, t, pest_extended, label="extended")
