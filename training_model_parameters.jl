include(joinpath(@__DIR__, "setup.jl"))

# Parameters
nstate = 1

# Second  order model
function wk4p(x, p, t)
    A, B = p
    A * x + B * [ϕc(t)]
end

function loss(ps, p)
    ns = p
    A = reshape(ps[1:ns^2], ns, ns)
    B = reshape(ps[(ns^2+1):(ns^2+ns)], ns, 1)
    C = reshape(ps[(ns^2+ns+1):(ns^2+2ns)], 1, ns)
    D = ps[(ns+1)^2]
    u0 = reshape(ps[((ns+1)^2+1):((ns+1)^2+ns)], ns)
    prob = ODEProblem(wk4p, u0, (0, tv[end]), (A, B))
    sol = solve(prob, saveat=h)
    pest = vec(C * Array(sol) + D * ϕc.(sol.t)')
    mean(abs2, pest - pc.(sol.t)), pest
end

# Setup random start
ps = rand(nstate^2+3nstate+1)

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())

optprob = OptimizationProblem(optf, ps, nstate)
optsol = solve(optprob, ADAM(0.05), maxiters=1000, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, nstate)
optsol = solve(optprob, ADAM(0.01), maxiters=200, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, nstate)
optsol = solve(optprob, BFGS(), callback=callback)

# Data and model to save
A = reshape(optsol.u[1:nstate^2], nstate, nstate)
B = reshape(optsol.u[(nstate^2+1):(nstate^2+nstate)], nstate, 1)
C = reshape(optsol.u[(nstate^2+nstate+1):(nstate^2+2nstate)], 1, nstate)
D = optsol.u[(nstate+1)^2]
u0 = reshape(optsol.u[((nstate+1)^2+1):((nstate+1)^2+nstate)], nstate)

prob = ODEProblem(wk4p, u0, (0, tv[end]), (A, B))
sol = solve(prob, saveat=h)
x = Array(sol)
dx = A * x + B * ϕc.(sol.t)'
p = C * x + D * ϕc.(sol.t)'



# Save data
println("Saving data")
writedlm(joinpath("data", "estimates", "order_$(nstate)_param_fit.csv"), [p' dx' x'], ',')
writedlm(joinpath("data", "estimates", "order_$(nstate)_params.csv"), optsol.u, ',')


# Plotting
p1 = plot(; xlabel="Time [s]", ylabel="Pressure [mmHg]")
plot!(p1, tv, pc.(tv), label="standard")
plot!(p1, tv, p', label="linear fit")
savefig(p1, joinpath("fig", "linear_parameter_fit.png"))