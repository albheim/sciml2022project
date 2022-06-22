include(joinpath(@__DIR__, "setup.jl"))

# Params
nstate = 1

# Read data
A = readdlm(joinpath("data", "estimates", "order_$(nstate)_nn.csv"), ',')
p = A[:, 1:1]'
dx = A[:, 2:nstate+1]'
x = A[:, nstate+2:2nstate+2]'

# Do symbolic regression, pick one at a time
prob_dyn = DirectDataDrivenProblem(x, dx)
prob_obs = DirectDataDrivenProblem(x, p)

# Same basis for both
@parameters t
@variables u[1:nstate](t) ϕ(t)
basis = Basis([
    polynomial_basis([u; ϕ], 2);
    #exp.(polynomial_basis(u, 2));
    #sin.(polynomial_basis([u; ϕ], 2));
], [u; ϕ])
println(basis)

res_dyn = solve(prob_dyn, basis, STLSQ(1.0))
res_obs = solve(prob_obs, basis, STLSQ(1.0))
println(result(res_dyn))
println(result(res_obs))

# Generate function
expr = substitute([Num(eq.rhs) for eq in equations(result(res_dyn))], Dict(parameter_map(res_dyn)))
fdyn, _ = build_function(expr, states(result(res_dyn)); expression=Val(false))
expr = substitute([Num(eq.rhs) for eq in equations(result(res_obs))], Dict(parameter_map(res_obs)))
fobs, _ = build_function(expr, states(result(res_obs)); expression=Val(false))

function wk4p_extended(x, p, t)
    f([x; ϕc(t)])
end

# Plotting
p1 = plot(tv, pc.(tv), label="standard")

plot!(p1, tv, p', label="nn")

prob_extended = ODEProblem(wk4p_extended, x[:, 1], (0, tv[end]))
sol_extended = solve(prob_extended, saveat=h)
pest_extended = fobs.(sol_extended.u, ϕc.(sol_extended.t))
plot!(p1, tv, pest_extended, label="symbolic regression")