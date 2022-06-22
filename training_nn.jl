include(joinpath(@__DIR__, "setup.jl"))

nstate = 2

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
    solve(prob, saveat=h)
end

function loss_nn(ps, p)
    nnsize1, nnsize2, = p
    sol = simulate_nn(ps, p)
    nnps = ps[1:nnsize2]
    pest = vec(re2(nnps)([Array(sol); ϕc.(sol.t)']))
    mean(abs2, pest - pc.(sol.t)), pest
end

ps = [nnps1; nnps2; rand(nstate)]
p = (length(nnps1), length(nnps2), nstate)

# Is this faster with Zygote?
optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

println("Running ADAM opt")
optprob = OptimizationProblem(optf, ps, p)
optsol = solve(optprob, ADAM(0.05), maxiters=1000, callback=callback)

println("Running BFGS opt")
optprob = OptimizationProblem(optf, optsol.u, p)
optsol = solve(optprob, BFGS(), callback=callback)

# Generate data for NN1
sol = simulate_nn(optsol.u, p)
x = [Array(sol); ϕc.(sol.t)']
dx = re1(optsol.u[1:length(nnps1)])(x)
p = re2(optsol.u[length(nnps1)+1:length(nnps1)+length(nnps2)])(x)

# Save data
println("Saving data")
writedlm(joinpath("data", "estimates", "order_$(nstate)_only_nn.csv"), [p' dx' x'], ',')
