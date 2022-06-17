using LinearAlgebra, DifferentialEquations, SparseArrays, Polynomials, Plots

struct PiecewisePolynomial{T}
    x::Vector{T}
    p::Vector{Polynomial{T,:x}}
end

struct CubicSpline{T}
    # FIXME: parametric type so CubicSpline is same type as its pwp element type
    pwp::PiecewisePolynomial{T}
    y # Would-be knot obscissas in absence of smoothing
    α # Smoothing parameter in (0,1)
    w # Weight scalar or vector
    #Js # Expected prediction error from leave-one-out cross validation
    periodic::Bool # Periodic extrapolation (as opposed to no extrapolation) 
end

struct CubicSplineProduct{T}
    s1::CubicSpline{T}
    s2::CubicSpline{T}
end

"Evaluate piecewise polynomial. Asserts x lies within closed interval defined by x[1] and x[end]"
function (pwp::PiecewisePolynomial)(x;der::Int=0)  # Type so PiecewisePolynomial has same type as x
    @assert pwp.x[1]<=x<=pwp.x[end]
    k=max(searchsortedfirst(pwp.x,x,lt=<)-1,1)  
    p=derivative(pwp.p[k],der)
    return p(x-pwp.x[k])  
end

"""Fit a possibly smoothed and possibly periodic cubic spline to points (x,y).
No smoothing if α=1 and linear if α=0. Amount of smoothing per point is determined by w."""
function CubicSpline(x::Vector{T},y::Vector{};periodic::Bool=true,α=1,w::Vector{T}=ones(T,length(x)-1))::CubicSpline{T} where T
    # Notation (almost) according to Zanetti "Periodic cubic smoothing splines as a quadratic minimization problem"
    n=length(x)
    h=diff(x)
    @assert n>=3
    @assert ~(0 in h)
    @assert length(w)==n-1
    
    # FIXME: should be able to store only one half of S,V since symmetric
    S=spdiagm(-1=>h[1:end-1],0=>2*(h+circshift(h,1)),1=>h[1:end-1])
    hi=1 ./h
    V=spdiagm(-1=>hi[1:end-1],0=>-hi-circshift(hi,1),1=>hi[1:end-1])
    if periodic # C2-periodicity conditions
        S[1,n-1]=h[end]=S[n-1,1]=h[end]    
        V[1,n-1]=V[n-1,1]=hi[end]
    else # Second derivatives at end point 0
        S[1,2]=S[n-1,n-2]=0
        V[1,1:2]=[0,0]
    end
    Q=ldlt(S)\V
    
    # Setup problem
    W=Diagonal(w) # FIXME: Doesn't deal with Infs in w (that arise from σ=0 for some measurements)
    U=2α*W+12*(1-α)*V'*Q # α corresponds to p in paper
    v=2α*W*y[1:end-1];

    # Solve for polynomial coefficients
    a=U\v
    c=3Q*a
    d=hi.*(circshift(c,-1)-c)./3
    b=hi.*(circshift(a,-1)-a)-c.*h-d.*h.^2
    
    #Σ=2α*inv(Matrix(U))
    #@show Σ*y[1:end-1] # Works but we should add final row that maps y[end] to last knot point

    # Store polynomials
    ps=[Polynomial([a[k],b[k],c[k],d[k]]) for k=1:n-1]
    pwp=PiecewisePolynomial(x,ps)
    return CubicSpline(pwp,y,α,w,periodic)
end

"Compute cubic spline (derivative) interpolation with scalar or vector argument."
function (s::CubicSpline)(x;der::Int=0)
    if s.periodic
        x=fold(x,s.pwp.x[1],s.pwp.x[end]) 
    else
        @assert s.pwp.x[1]<=x<=s.pwp.x[end]
    end
    return s.pwp(x,der=der)
end

import Base.*
"Define multiplication for CubicSpline"
s1::CubicSpline * s2::CubicSpline = CubicSplineProduct(s1,s2)

"Evaluate (derivative of) CubicSplineProduct"
function (sp::CubicSplineProduct)(x;der::Int=0)
    ret=0
    for k in 0:der
       ret+=binomial(der,k)*sp.s1(x,der=k)*sp.s2(x,der=der-k) # terms of the product chain rule
    end
    return ret
end

# Vectorized versions
# function (s::Union{CubicSpline,CubicSplineProduct})(x::Vector;der::Int=0)
#     return s.(x,der=der)
# end

"Periodically fold back value to range"
function fold(x,xmin,xmax)
    Δx=xmax-xmin
    x=(x-xmin) % Δx
    x=x<0 ? Δx+x : x;
    x+=xmin
end