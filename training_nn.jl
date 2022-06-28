include(joinpath(@__DIR__, "setup.jl"))

# Neural net 
hidden = 16
nnps1, re1 = Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, Flux.elu),
        Dense(hidden, hidden, Flux.elu),
        Dense(hidden, nstate)
    )
)
nnps2, re2 = Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, Flux.elu),
        Dense(hidden, hidden, Flux.elu),
        Dense(hidden, 1)
    )
)

function wk4p_nn(x, p, t)
    vec(re1(p)([x; ϕc(t)]))
end

function simulate_nn(ps, p)
    nnsize1, nnsize2, nstate = p
    nnps = ps[nnsize2+1:nnsize2+nnsize1]
    u0 = ps[end-nstate+1:end]
    prob = ODEProblem(wk4p_nn, u0, (0, tv[end]), nnps)
    solve(prob, saveat=tv)
end

function loss_nn(ps, p)
    nnsize1, nnsize2, = p
    sol = simulate_nn(ps, p)
    nnps = ps[1:nnsize2]
    pest = vec(re2(nnps)([Array(sol); ϕc.(sol.t)']))
    mean(abs2, pest - pc.(sol.t)), pest
end

ps = [nnps2; nnps1; rand(nstate)]
p = (length(nnps1), length(nnps2), nstate)

optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

println("Running ADAM opt")
optprob = OptimizationProblem(optf, ps, p)
optsol = solve(optprob, ADAM(0.1), maxiters=200, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, ADAM(0.01), maxiters=50, callback=callback)

println("Running BFGS opt")
optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, BFGS(), callback=callback)

# Generate data 
sol = simulate_nn(optsol.u, p)
x = [Array(sol); ϕc.(sol.t)']
dx = re1(optsol.u[length(nnps2)+1:length(nnps2)+length(nnps1)])(x)
pest = re2(optsol.u[1:length(nnps2)])(x)

# Save data
println("Saving data")
writedlm(joinpath("data", "estimates", "order_$(nstate)_nn.csv"), [pest' dx' x'], ',')
writedlm(joinpath("data", "estimates", "order_$(nstate)_nn_params.csv"), optsol.u, ',')

# Read data
data = readdlm(joinpath("data", "estimates", "order_$(nstate)_nn.csv"), ',')
pest = data[:, 1:1]'
dx = data[:, 2:nstate+1]'
x = data[:, nstate+2:2nstate+2]'

# First we find dynamics function
prob = DirectDataDrivenProblem(x, dx)

@parameters t
@variables u[1:nstate](t) ϕ(t)
basis = Basis([
    polynomial_basis([u; ϕ], 2);
    #exp.(polynomial_basis(u, 2));
    #sin.(polynomial_basis([u; ϕ], 2));
], [u; ϕ])
println(basis)

λs = nstate == 1 ? exp10.(-2:0.1:1) : exp10.(-1:0.1:2)
opt = STLSQ(λs)

res = solve(prob, basis, opt)
println(result(res))
plot(res)

# Generate function
expr_dyn = substitute([Num(eq.rhs) for eq in equations(result(res))], Dict(parameter_map(res)))
f_dyn, = build_function(expr_dyn, states(result(res)); expression=Val(false))

wk4p_extended(x, p, t) = p([x; ϕc(t)])
prob = ODEProblem(wk4p_extended, x[1:end-1, 1], (0, tv[end]), f_dyn)
sol = solve(prob, saveat=tv)
x_ext = [Array(sol); ϕc.(sol.t)']

# Then we find observation function, based on new dynamics function
prob = DirectDataDrivenProblem(x_ext, pest)

λs = exp10.(-2:0.1:1) 
opt = STLSQ(λs)

res = solve(prob, basis, opt)
println(result(res))
plot(res)

# Generate function
expr_obs = substitute([Num(eq.rhs) for eq in equations(result(res))], Dict(parameter_map(res)))
f_obs, = build_function(expr_obs, states(result(res)); expression=Val(false))

p_ext = [f_obs(c)[1] for c in eachcol(x_ext)]'

open(joinpath("data", "estimates", "order_$(nstate)_nn_symbolic_fit.txt"), "w") do io
    println(io, expr_dyn)
    println(io, expr_obs)
end

# Plotting
p1 = plot(; xlabel="Time [s]", ylabel="Pressure [mmHg]", title="$(nstate) state model, learned dynamics and observations")

scatter!(p1, tv, pc.(tv), label="data")

wkpar = get_standard_model(nstate)
prob = ODEProblem(wk4p, wkpar[5], (0, tv[end]), wkpar[1:2])
sol = solve(prob, saveat=tv)
p_wk = wkpar[3] * Array(sol) + wkpar[4] * ϕc.(sol.t)'
mse_wk = sum(abs2, p_wk' - pc.(tv)) / length(tv)
plot!(p1, tv, p_wk', label="wk")

mse_nn = sum(abs2, pest' - pc.(tv)) / length(tv)
plot!(p1, tv, pest', label="nn")

mse_symb = sum(abs2, p_ext' - pc.(tv)) / length(tv)
plot!(p1, tv, p_ext', label="symbolic regression")

savefig(p1, joinpath("fig", "order_$(nstate)_nn.png"))

writedlm(joinpath("data", "estimates", "order_$(nstate)_nn_mse.csv"), [mse_wk, mse_nn, mse_symb])
