include(joinpath(@__DIR__, "setup.jl"))

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

############### FIRST ORDER MODEL #############################3
function wk4p!(dx, x, p, t)
    @. dx = p[1] * x + p[2] * ϕc(t)
end

function loss(ps, p)
    A, B, C, D, u0 = ps
    prob = ODEProblem(wk4p!, [u0], (0, tv[end]), (A, B))
    sol = solve(prob, saveat=h)
    pest = C * sol[1, :] + D * ϕc.(sol.t)
    mean(abs2, pest .- pc.(sol.t)), pest
end

ps = Float64[-1, 1, 1, 1, 1]

optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, ps)
optsol = solve(optprob, ADAM(0.1), maxiters=1000, callback=callback)


optpar = optsol.u

hidden = 32
nn = Chain(
    Dense(2, hidden, tanh), 
    Dense(hidden, hidden, tanh), 
    Dense(hidden, 1)
)
nnps, re = Flux.destructure(nn)


function wk4p_nn!(dx, x, p, t)
    A, B, nnps = p
    ϕ = ϕc(t)
    dx .= A * x .+ B * ϕ .+ re(nnps)([x; ϕ])
end

function simulate(ps, p)
    nnlen, = p
    A = ps[1]
    B = ps[2] 
    u0 = ps[5]
    prob = ODEProblem(wk4p_nn!, [u0], (0, tv[end]), (A, B, ps[6:5+nnlen]))
    solve(prob, saveat=h)
end

function loss_nn(ps, p)
    nnlen, = p
    nnps = ps[(6+nnlen):end]
    C = ps[3]
    D = ps[4]
    sol = simulate(ps, p)
    x = Array(sol)
    ϕ = ϕc.(sol.t)
    pest = vec(C * x + D * ϕ + re(nnps)([x; ϕ]))
    mean(abs2, pest .- pc.(sol.t)), pest
end

ps = [optpar; nnps]

optf = OptimizationFunction(loss_nn, Optimization.AutoForwardDiff())

optprob = OptimizationProblem(optf, ps, (length(nnps),))
optsol = solve(optprob, ADAM(0.05), maxiters=100, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, (length(nnps),))
optsol = solve(optprob, ADAM(0.01), maxiters=100, callback=callback)

optprob = OptimizationProblem(optf, optsol.u, (length(nnps),))
optsol = solve(optprob, BFGS(), callback=callback)

# Do symbolic regression
sol = simulate(optsol.u, ())
x = [Array(sol); ϕc.(sol.t)']
y = re(optsol.u[6:end])(x)

prob = DirectDataDrivenProblem(x, y)

@variables u[1:2]
basis = Basis([
    polynomial_basis([u[1]; u[2]], 4);
    exp.(polynomial_basis([u[1]; u[2]], 4));
    sin.(polynomial_basis([u[1]; u[2]], 4));
], u)
println(basis)

res = solve(prob, basis, STLSQ())
println(res)
println(result(res))

# Check solution
eq = substitute(equations(result(res))[1].rhs, parameter_map(res))
generate_function(eq, u; expression=Val(true))

# Should be possible to automatically generate this
f(x) = 2.7*sin(x[1]) + 0.2*x[1]*x[2] - 2.6*x[2]

plot(y')
plot!([f(xx) for xx in eachcol(x)])

function wk4p_extended!(dx, x, p, t)
    wk4p!(dx, x, p, t) # Update according to base eq
    dx .+= f([x; ϕc(t)])
end

A, B, C, D, u0 = ps[1:5]

p1 = plot(tv, pc.(tv), label="real")

prob_default = ODEProblem(wk4p!, [u0], (0, tv[end]), (A, B))
sol_default = solve(prob_default, saveat=h)
pest_default = C * sol_default[1, :] + D * ϕc.(sol_default.t)
plot!(p1, tv, pest_default, label="default")

prob_nn = ODEProblem(wk4p_nn!, [u0], (0, tv[end]), (A, B, ps[6:end]))
sol_nn = solve(prob_nn, saveat=h)
pest_nn = C * sol_nn[1, :] + D * ϕc.(sol_nn.t)
plot!(p1, tv, pest_nn, label="nn")

prob_extended = ODEProblem(wk4p_extended!, [u0], (0, tv[end]), (A, B))
sol_extended = solve(prob_extended, saveat=h)
pest_extended = C * sol_extended[1, :] + D * ϕc.(sol_extended.t)
plot!(p1, tv, pest_extended, label="extended")