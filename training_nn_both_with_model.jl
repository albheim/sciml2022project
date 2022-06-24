include(joinpath(@__DIR__, "setup.jl"))

# Parameters
nstate = 2

# Neural net fit
hidden = 16
nnps_dyn, re_dyn = Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, Flux.elu),
        Dense(hidden, hidden, Flux.elu),
        Dense(hidden, nstate)
    )
)
nnps_obs, re_obs = Flux.destructure(
    Chain(
        Dense(nstate + 1, hidden, Flux.elu),
        Dense(hidden, hidden, Flux.elu),
        Dense(hidden, 1)
    )
)

function wk4p(x, p, t)
    A, B = p
    A * x + B * [ϕc(t)]
end

function wk4p_nn(x, p, t)
    A = reshape(p[1:nstate^2], nstate, nstate)
    B = reshape(p[(nstate^2+1):(nstate^2+nstate)], nstate, 1)
    nnps = p[(nstate^2+nstate+1):end]
    vec(wk4p(x, (A, B), t) + re_dyn(nnps)([x; ϕc(t)]))
end


function loss_nn(ps, p)
    A, B, C, D, u0, nparams_dyn = p
    psall = [vec(A); vec(B); ps[1:nparams_dyn]]
    prob = ODEProblem(wk4p_nn, u0, (0, tv[end]), psall)
    sol = solve(prob, saveat=h)
    nnps = ps[nparams_dyn+1:end]
    x = Array(sol)
    ϕ = ϕc.(sol.t)'
    pest = vec(C * x + D * ϕ + re_obs(nnps)([x; ϕ]))
    mean(abs2, pest .- pc.(sol.t)), pest
end

ps = [nnps_dyn; nnps_obs]
p = (get_standard_model(nstate)..., length(nnps_dyn))

# Is this faster with Zygote?
optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

optprob = OptimizationProblem(optf, ps, p)
optsol = solve(optprob, ADAM(0.001), maxiters=500, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, BFGS(), callback=callback)

# Generate data
psall = [vec(A); vec(B); ps]
prob = ODEProblem(wk4p_nn, u0, (0, tv[end]), psall)
sol = solve(prob, saveat=h)
x = [Array(sol); ϕc.(sol.t)']
nnout = re(optsol.u)(x)
dx = wk4p.(sol.u, ((A, B),), sol.t)

# Save data
println("Saving data")
writedlm(joinpath("data", "estimates", "order_$(nstate)_nn_with_model.csv"), [p' dx' x'], ',')


# Do symbolic regression
prob = DirectDataDrivenProblem(x, y)

@variables u[1:2] ϕ
basis = Basis([
    polynomial_basis([u; ϕ], 3);
    #sin.(polynomial_basis([u; ϕ], 2));
], [u; ϕ])
println(basis)

res = solve(prob, basis, STLSQ(2.0))
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

p1 = plot(tv, pc.(tv), label="real")

prob_default = ODEProblem(wk4p, [u0], (0, tv[end]), (A, B))
sol_default = solve(prob_default, saveat=h)
pest_default = C * sol_default[1, :] + D * ϕc.(sol_default.t)
plot!(p1, tv, pest_default, label="default")

prob_nn = ODEProblem(wk4p_nn, [u0], (0, tv[end]), (A, B, ps[6:end]))
sol_nn = solve(prob_nn, saveat=h)
pest_nn = C * sol_nn[1, :] + D * ϕc.(sol_nn.t)
plot!(p1, tv, pest_nn, label="nn")

prob_extended = ODEProblem(wk4p_extended!, [u0], (0, tv[end]), (A, B))
sol_extended = solve(prob_extended, saveat=h)
pest_extended = C * sol_extended[1, :] + D * ϕc.(sol_extended.t)
plot!(p1, tv, pest_extended, label="extended")
