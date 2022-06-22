function first_order_standard_model()
    1, 1, 1, 1, 1 # TODO
end

function second_order_standard_model()
    Rp = 0.79 # Peripheral (systemic) resistance [mmHg*ml^-1*s]
    Cc = 1.22 # Total arterial compliance [ml/mmHg]
    Rc = 0.056 # characeristic aoritic resistance
    L = 0.0051 # Total arterial impedance [mmHg*s^2*ml^-1]
    Rp, Cc, Rc, L = [1000 / 60, 60 / 1000, 1000 / 60, 1000 / 60] .* [Rp, Cc, Rc, L] # convert impedance units to match L/min flows 

    A = [-1/(Cc*Rp) 0; 0 -Rc/L]
    B = [1; Rc]
    C = [1 / Cc -Rc / L]
    D = Rc
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