function first_order_standard_model()
    Rp = 13.6
    Cc = 0.0996

    A = [-1/(Cc*Rp);;]
    B = [1.0;;]
    C = [1/Cc;;]
    D = [0.0;;]
    u0 = [7.20111]

    A, B, C, D, u0
end

function second_order_standard_model()
    Rp = 0.79 # Peripheral (systemic) resistance [mmHg*ml^-1*s]
    Cc = 1.22 # Total arterial compliance [ml/mmHg]
    Rc = 0.056 # characeristic aoritic resistance
    L = 0.0051 # Total arterial impedance [mmHg*s^2*ml^-1]
    Rp, Cc, Rc, L = [1000 / 60, 60 / 1000, 1000 / 60, 1000 / 60] .* [Rp, Cc, Rc, L] # convert impedance units to match L/min flows 

    A = [-1/(Cc*Rp) 0; 0 -Rc/L]
    B = [1; Rc;;]
    C = [1/Cc -Rc/L]
    D = [Rc;;]
    u0 = [4.82090092759144, 0.04580860132564981]

    A, B, C, D, u0
end

function get_standard_model(order)
    if order == 1
        first_order_standard_model()
    else
        second_order_standard_model()
    end
end

function wk4p(x, p, t)
    A, B = p
    A * x + B * [ϕc(t)]
end

function plot_standard()
    A, B, C, D, u0 = get_standard_model(1)
    prob = ODEProblem(wk4p, u0, (0, tv[end]), (A, B))
    sol = solve(prob, saveat=h)
    x = Array(sol)
    p1 = C * x + D * ϕc.(sol.t)'

    A, B, C, D, u0 = get_standard_model(2)
    prob = ODEProblem(wk4p, u0, (0, tv[end]), (A, B))
    sol = solve(prob, saveat=h)
    x = Array(sol)
    p2 = C * x + D * ϕc.(sol.t)'

    pp = plot(tv, [pc.(tv) p1' p2'], right_margin=5Plots.mm,legend=:topleft,ylabel="Pressure [mmHg]",xlabel="Time [s]",labels=["p" "p̂₁" "p̂₂"],size=0.8.*(600,400),ylims=[60,120])
    plot!(twinx(), tv, ϕc.(tv),ylims=[-5,70])
    pϕ = plot(tv, ϕc.(tv))
    display(pp,pϕ,layout=(2,1))
end
