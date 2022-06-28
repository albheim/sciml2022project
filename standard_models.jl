function first_order_standard_model(params=[13.6, 0.0996])
    Rp, Cc = params
    A = [-1/(Cc*Rp);;]
    B = [1.0;;]
    C = [1/Cc;;]
    D = [0.0;;]
    u0 = [7.20111]

    A, B, C, D, u0
end


function second_order_standard_model(params=[0.79, 1.22, 0.056, 0.0051])
    Rp, Cc, Rc, L = [1000 / 60, 60 / 1000, 1000 / 60, 1000 / 60] .* params

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
