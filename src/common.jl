# common facilities

# tools to check size

function nmf_checksize(X, W::AbstractMatrix, H::AbstractMatrix)

    p = size(X, 1)
    n = size(X, 2)
    k = size(W, 2)

    if !(size(W,1) == p && size(H) == (k, n))
        throw(DimensionMismatch("Dimensions of X, W, and H are inconsistent."))
    end

    return (p, n, k)
end


# the result type

struct Result{T}
    W::Matrix{T}
    H::Matrix{T}
    niters::Int
    converged::Bool
    objvalue::T
    objvalues::Vector{T}
    sparsevalues::Vector{T}
    avgfits::Vector{T}

    function Result{T}(W::Matrix{T}, H::Matrix{T}, niters::Int, converged::Bool, objv, objvs, sparsevalues,avgfits) where T
        if size(W, 2) != size(H, 1)
            throw(DimensionMismatch("Inner dimensions of W and H mismatch."))
        end
        new{T}(W, H, niters, converged, objv, objvs, sparsevalues,avgfits)
    end
end


Base.:(==)(A::Result, B::Result) = A.W == B.W && A.H == B.H && A.niters == B.niters && A.converged == B.converged && A.objvalue == B.objvalue
Base.hash(s::Result, h::UInt) = hash(s.objvalue, hash(s.converged, hash(s.niters, hash(s.H, hash(s.W, h + (0x09c9f08cfcba6de3 % UInt))))))


# common algorithmic skeleton for iterative updating methods

abstract type NMFUpdater{T} end
evaluate_sparseness(updater::NMFUpdater{T}, state, X, W, H) where T = zero(T)

function nmf_skeleton!(updater::NMFUpdater{T},
                       X, W::Matrix{T}, H::Matrix{T},
                       maxiter::Int, verbose::Bool, tol;
                       U::Matrix{T}=Matrix{T}(undef,0,0),
                       Vt::Matrix{T}=Matrix{T}(undef,0,0),
                       d::Vector{T}=Vector{T}(undef,0),
                       gtW::Matrix{T}=Matrix{T}(undef,0,0),
                       gtH::Matrix{T}=Matrix{T}(undef,0,0),
                       maskW::Union{Colon,Vector,BitVector}=Colon(),
                       maskH::Union{Colon,Vector,BitVector}=Colon()
                       ) where T
    objv = convert(T, NaN)

    # init
    state = prepare_state(updater, X, W, H, U=U, Vt=Vt, d=d, gtW=gtW[maskW,:], gtH=gtH[:,maskH])
    preW = Matrix{T}(undef, size(W))
    preH = Matrix{T}(undef, size(H))
    objvs = T[]; objvsparses = T[]; avgfits=T[]
    if verbose
        start = time()
        objv = evaluate_objv(updater, state, X, W, H)
        push!(objvs,objv)
        push!(objvsparses,evaluate_sparseness(updater, state, X, W, H))
        push!(avgfits,evaluate_fitvalue(updater, state, X[maskW,maskH], W[maskW,:], H[:,maskH]))
        # @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(W & H).change")
        # @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
    end

    # main loop
    converged = false
    t = 0
    while !converged && t < maxiter
        t += 1
        copyto!(preW, W)
        copyto!(preH, H)

        # update H
        update_wh!(updater, state, X, W, H)

        # determine convergence
        converged, dev = stop_condition(W, preW, H, preH, tol)

        # display info
        if verbose
            elapsed = time() - start
            preobjv = objv
            objv = evaluate_objv(updater, state, X, W, H)
            push!(objvs,objv)
            push!(objvsparses,evaluate_sparseness(updater, state, X, W, H))
            push!(avgfits,evaluate_fitvalue(updater, state, X[maskW,maskH], W[maskW,:], H[:,maskH]))
            #@printf("%5d    %13.6e    %13.6e    %13.6e    %13.6e\n",
            #    t, elapsed, objv, objv - preobjv, dev)
        end
    end

    if !verbose
        objv = evaluate_objv(updater, state, X, W, H)
    end
    return Result{T}(W, H, t, converged, objv, objvs, objvsparses, avgfits)
end


function stop_condition(W::AbstractArray{T}, preW::AbstractArray, H::AbstractArray, preH::AbstractArray, eps::AbstractFloat) where T
    devmax = zero(T)
    for j in axes(W,2)
        dev_w = sum_w = zero(T)
        for i in axes(W,1)
            dev_w += (W[i,j] - preW[i,j])^2
            sum_w += (W[i,j] + preW[i,j])^2
        end
        dev_h = sum_h = zero(T)
        for i in axes(H,2)
            dev_h += (H[j,i] - preH[j,i])^2
            sum_h += (H[j,i] + preH[j,i])^2
        end
        devmax = max(devmax, sqrt(max(dev_w/sum_w, dev_h/sum_h)))
        if sqrt(dev_w) > eps*sqrt(sum_w) || sqrt(dev_h) > eps*sqrt(sum_h)
            return false, devmax
        end
    end
    return true, devmax
end
