# Coordinate descent method, translated from the Python/Cython implementation
#  in scikit-learn and modified to comply with the interfaces of the NMF package

# Original files
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/nmf.py
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/cdnmf_fast.pyx

# Original implementation authors:
# Vlad Niculae
# Lars Buitinck
# Mathieu Blondel <mathieu@mblondel.org>
# Tom Dupre la Tour

# Original license: BSD 3 clause

# Julia translation: Vilim Štih

# Reference: Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
#  large scale nonnegative matrix and tensor factorizations."
#  IEICE transactions on fundamentals of electronics, communications and
#  computer sciences 92.3: 708-721, 2009.


mutable struct CoordinateDescent{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on W and H upon convergence
    update_H::Bool         # whether to update H
    α::T                   # constant that multiplies the regularization terms
    l₁ratio::T             # select whether the regularization affects the components (H), 
                           # the transformation (W), both or none of them 
                           # (:components, :transformation, :both, :none)
    regularization::Symbol # l1 / l2 regularization mixing parameter (in [0; 1])
    shuffle::Bool          # # if true, randomize the order of coordinates in the CD solver
    PCB_penmetric::Symbol  # PCB add. :HALS, :PCB
    PCB_αw::T
    PCB_αh::T

    function CoordinateDescent{T}(;maxiter::Integer=100,
                              verbose::Bool=false,
                              tol::Real=cbrt(eps(T)),
                              update_H::Bool=true,
                              α::Real=zero(T),
                              regularization=:both,
                              l₁ratio::Real=zero(T),
                              shuffle::Bool=false,
                              PCB_penmetric::Symbol=:HALS,
                              PCB_αw::Real=100,
                              PCB_αh::Real=100) where T
        new{T}(maxiter, verbose, tol, update_H, α, l₁ratio, regularization, shuffle, PCB_penmetric,
                    PCB_αw, PCB_αh)
    end
end


solve!(alg::CoordinateDescent{T}, X, W, H; U::Matrix{T}=Matrix{T}(undef,0,0), Vt::Matrix{T}=Matrix{T}(undef,0,0),
        d::Vector{T}=Vector{T}(undef,0), gtW::Matrix{T}=Matrix{T}(undef,0,0), gtH::Matrix{T}=Matrix{T}(undef,0,0),
        maskW::Union{Colon,Vector,BitVector}=Colon(),maskH::Union{Colon,Vector,BitVector}=Colon()) where {T} =
    nmf_skeleton!(CoordinateDescentUpd{T}(alg.α, alg.l₁ratio, alg.regularization, alg.shuffle, alg.update_H,
            alg.PCB_penmetric, alg.PCB_αw, alg.PCB_αh), X, W, H, alg.maxiter, alg.verbose, alg.tol;
            U=U, Vt=Vt, d=d, gtW=gtW, gtH=gtH, maskW=maskW, maskH=maskH)

struct CoordinateDescentUpd{T} <: NMFUpdater{T}
    l₁W::T
    l₂W::T
    l₁H::T
    l₂H::T
    shuffle::Bool
    update_H::Bool
    PCB_penmetric::Symbol
    PCB_αw::T
    PCB_αh::T
    function CoordinateDescentUpd{T}(α::T, l₁ratio::T, regularization::Symbol, shuffle::Bool, update_H::Bool,
                PCB_penmetric::Symbol,PCB_αw::T, PCB_αh::T) where {T}
        αW = zero(T)
        αH = zero(T)

        if (regularization == :both) || (regularization == :components)
            αH = α
        end

        if (regularization == :both) || (regularization == :transformation)
            αW = α
        end

        new{T}(αW*l₁ratio,
               αW*(1-l₁ratio),
               αH*l₁ratio,
               αH*(1-l₁ratio),
               shuffle,
               update_H,
               PCB_penmetric,
               PCB_αw,
               PCB_αh)
    end
end

mutable struct CoordinateDescentState{T}
    HHt::Matrix{T}
    XHt::Matrix{T}
    XtW::Matrix{T}
    violation::T
    violation_init::Union{Nothing, T}
    U::Matrix{T}
    Vt::Matrix{T}
    d::Vector{T}
    gtW::Matrix{T}
    gtH::Matrix{T}
    normW::T
    normH::T

    function CoordinateDescentState{T}(X, W, H, U, Vt, d, gtW, gtH, violation, violation_init) where T
        p, n, k = nmf_checksize(X, W, H)
        new{T}(Matrix{T}(undef, k, k),
               Matrix{T}(undef, p, k),
               Matrix{T}(undef, n, k),
               violation,
               violation_init,
               U,
               Vt,
               d,
               gtW,
               gtH,
               norm(W,1),
               norm(H,1)
               )
    end
end

prepare_state(::CoordinateDescentUpd{T}, X, W, H;
        U::Matrix{T}=Matrix{T}(undef,0,0), Vt::Matrix{T}=Matrix{T}(undef,0,0), d::Vector{T}=Vector{T}(undef,0),
        gtW::Matrix{T}=Matrix{T}(undef,0,0), gtH::Matrix{T}=Matrix{T}(undef,0,0)
        ) where T = CoordinateDescentState{T}(X, W, H, U, Vt, d, gtW, gtH, zero(T), nothing)

function evaluate_objv(updater::CoordinateDescentUpd{T}, s::CoordinateDescentState{T}, X, W, H) where T
    # convert(T, 0.5) * sqL2dist(X, s.WH)
    if updater.PCB_penmetric ∈ [:HALS, :SPARSE_W, :SPARSE_H]
        sqL2dist(X, W*H)
    elseif updater.PCB_penmetric == :PCB
        M = s.U\W; N = H/s.Vt
        sqL2dist(Diagonal(s.d), M*N)
    end
end
function evaluate_sparseness(updater::CoordinateDescentUpd{T}, s::CoordinateDescentState{T}, X, W, H) where T
    if updater.PCB_penmetric == :HALS
        updater.l₁W*norm(W,1) + updater.l₂W*norm(W)^2 + updater.l₁H*norm(H,1) + updater.l₂H*norm(H)^2
    elseif updater.PCB_penmetric == :PCB
        M = s.U\W; N = H/s.Vt
        normwp = norm(s.U,1); normhp = norm(s.Vt,1); (αw, αh) = (updater.PCB_αw/normwp, updater.PCB_αh/normhp)
        αw*norm(s.U*M,1) + αh*norm(N*s.Vt,1)
    elseif updater.PCB_penmetric == :SPARSE_W
        Wn, Hn = copy(W), copy(H); normalizeW!(Wn,Hn)
        norm(Wn,1)#/s.normW
    elseif updater.PCB_penmetric == :SPARSE_H
        norm(H,1)#/s.normH
    else
        zero(T)
    end
end
function evaluate_fitvalue(updater::CoordinateDescentUpd{T}, s::CoordinateDescentState{T}, X, W, H) where T
    if !isempty(s.gtW) && !isempty(s.gtH)
        avgfit, _ =  matchedfitval(s.gtW, s.gtH, W, H; clamp=false)
    else
        avgfit = fitd(X,W*H)
    end
    avgfit
end

"Updates W only"
function _update_coord_descent!(s::CoordinateDescentState{T}, X, W, H, 
                                l1_reg, l2_reg, shuffle::Bool, W_flag::Bool) where T
    Ht = transpose(H)
    HHt = s.HHt
    mul!(HHt, H, Ht)
    if W_flag
        XHt = s.XHt
    else
        XHt = s.XtW
    end
    mul!(XHt, X, Ht)

    n_components = size(H, 1)
    n_samples = size(W, 1)

    if l2_reg > 0.
        HHt[diagind(HHt)] .+= l2_reg
    end
    if l1_reg > 0.
        XHt .-= l1_reg
    end
    if shuffle
        permutation = randperm(n_components)
    else
        permutation = 1:n_components
    end

    violation = zero(eltype(X))

    for t in permutation
        for i in 1:n_samples
             # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
            grad = -XHt[i, t]

            for r in 1:n_components
                grad += HHt[t, r] * W[i, r]
            end

            # projected gradient
            pg = W[i, t] == 0 ? min(zero(grad), grad) : grad
            violation += abs(pg)

            # Hessian
            hess = HHt[t, t]
            if hess != 0
                W[i, t] = max(W[i, t] - grad / hess, zero(grad))
            end
        end
    end
    return violation
end


function update_wh!(upd::CoordinateDescentUpd{T}, s::CoordinateDescentState{T},
                    X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    violation = zero(T)

    # update W
    violation += _update_coord_descent!(s, X, W, H, upd.l₁W, upd.l₂W, upd.shuffle, true)

    # update H
    if upd.update_H
        Wt = transpose(W)
        Ht = transpose(H)
        Xt = transpose(X)
        violation += _update_coord_descent!(s, Xt, Ht, Wt, upd.l₁H, upd.l₂H, upd.shuffle, false)
    end

    s.violation = violation
    if s.violation_init !== nothing
        s.violation_init = violation
    end
end
