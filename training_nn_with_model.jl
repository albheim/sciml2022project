include(joinpath(@__DIR__, "setup.jl"))

# Parameters
nstate = 2

# Neural net fit
hidden = 16
nnps, re= Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, Flux.elu),
        Dense(hidden, hidden, Flux.elu),
        Dense(hidden, nstate)
    )
)

function wk4p_nn(x, p, t)
    A = reshape(p[1:nstate^2], nstate, nstate)
    B = reshape(p[(nstate^2+1):(nstate^2+nstate)], nstate, 1)
    nnps = p[(nstate^2+nstate+1):end]
    vec(wk4p(x, (A, B), t) + re(nnps)([x; ϕc(t)]))
end


function loss_nn(ps, p)
    A, B, C, D, u0 = p
    psall = [vec(A); vec(B); ps]
    prob = ODEProblem(wk4p_nn, u0, (0, tv[end]), psall)
    sol = solve(prob, saveat=h)
    pest = vec(C * Array(sol) + D * ϕc.(sol.t)')
    mean(abs2, pest .- pc.(sol.t)), pest
end

ps = nnps
p = get_standard_model(nstate)

# Is this faster with Zygote?
optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

println("Running ADAM opt")
optprob = OptimizationProblem(optf, ps, p)
optsol = solve(optprob, ADAM(0.001), maxiters=500, callback=callback)

println("Running BFGS opt")
optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, BFGS(), callback=callback)

# Generate data
psall = [vec(A); vec(B); optsol.u]
prob = ODEProblem(wk4p_nn, u0, (0, tv[end]), psall)
sol = solve(prob, saveat=h)
x = [Array(sol); ϕc.(sol.t)']
nnout = re(optsol.u)(x)
dx = reduce(hcat, wk4p.(sol.u, ((A, B),), sol.t))
pest = [C D] * x 

# Save data
println("Saving data")
writedlm(joinpath("data", "estimates", "order_$(nstate)_nn_with_model.csv"), [pest' dx' x' nnout'], ',')
writedlm(joinpath("data", "estimates", "order_$(nstate)_nn_with_model_params.csv"), optsol.u, ',')

# Read data
data = readdlm(joinpath("data", "estimates", "order_$(nstate)_nn_with_model.csv"), ',')
pest = data[:, 1:1]'
dx = data[:, 2:nstate+1]'
x = data[:, nstate+2:2nstate+1]'
ϕdata = data[:, 2nstate+2:2nstate+2]'
nnout = data[:, 2nstate+3:end]'

# Better approach?
prob = ContinuousDataDrivenProblem(x, dx, U=ϕdata)
plot(prob)

@parameters t
@variables u[1:nstate](t) ϕ(t)
basis = Basis(polynomial_basis([u; ϕ], 2), u, controls=[ϕ])
println(basis)


# Do symbolic regression, pick one at a time
prob = DirectDataDrivenProblem([x; ϕdata], nnout)

# Same basis for both
@parameters t
@variables u[1:nstate](t) ϕ(t)
basis = Basis([
    polynomial_basis([u; ϕ], 2);
    #exp.(polynomial_basis(u, 2));
    #sin.(polynomial_basis([u; ϕ], 2));
], [u; ϕ])
println(basis)

λs = exp10.(-10:0.1:-3)
opt = STLSQ(λs)
res = solve(prob, basis, opt)
println(result(res))

# Generate function
expr = substitute([Num(eq.rhs) for eq in equations(result(res))], Dict(parameter_map(res)))
open(joinpath("data", "estimates", "order_$(nstate)_nn_with_model_symbolic_fit.txt"), "w") do io
    println(io, expr)
end
state = states(result(res))
f, = build_function(expr, state; expression=Val(false))

function wk4p_extended(x, p, t)
    wk4p(x, p[1:2], t) + p[3](x, ϕc(t))
end

# Plotting
p1 = plot(; xlabel="Time [s]", ylabel="Pressure [mmHg]", title="$(nstate) state model using WK + NN dynamics")

scatter!(p1, tv, pc.(tv), label="data")

wkpar = get_standard_model(nstate)
prob = ODEProblem(wk4p, wkpar[5], (0, tv[end]), wkpar[1:2])
sol = solve(prob, saveat=h)
p_wk = wkpar[3] * Array(sol) + wkpar[4] * ϕc.(sol.t)'
plot!(p1, tv, p_wk', label="windkessel")

plot!(p1, tv, pest', label="nn")

prob = ODEProblem(wk4p_extended, x[1:nstate, 1], (0, tv[end]), (wkpar[1], wkpar[2], f))
sol = solve(prob, saveat=h)
p_ext = wkpar[3] * Array(sol) + wkpar[4] * ϕc.(sol.t)'
plot!(p1, tv, p_ext', label="symbolic regression")

savefig(p1, joinpath("fig", "order_$(nstate)_nn_with_model.png"))
