const RESTRICT_KERNEL = reshape(0.25^2*[1f0 2f0 1f0; 2f0 4f0 2f0; 1f0 2f0 1f0], 3, 3, 1, 1)

"""
    mg_tvd_pds(y, λ, L=1; kw...)

Inputs may be images, matries, or tensors.
"""
function mg_tvd_pds(y, args...; kw...)
	x, hist = mg_tvd_pds(img2tensor(y), args...; kw...)
	return tensor2img(x), hist
end

function mg_tvd_pds(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = mg_tvd_pds(y, args...; kw...)
	return x[:,:], hist
end

function mg_tvd_pds(y::Array{T,3}, λ, L=1; isotropic=true, ℓ1=false, α=0.1, θ=0, maxit=100, n_inner=1, n_coarse=100, tol=1e-5, verbose=true, linesearch=true, t=1) where T
	M, N, Channels = size(y)
	y = reshape(y, size(y)...,1)  # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension

    # min_x max_z (1/2)||x-y||^2 + z^TDx - i_{λMaxBall}(z)
    η = 0.99 / sqrt(8)
    σ = 0.99 / sqrt(8)

	# initialization
	O = zeros(maxit);   # store objective fun history
	r = zeros(maxit);   # store primal residual

	W, Wᵀ = fdkernel(T)
	D(x) = conv(pad_circular(x, (1,0,1,0)), W);
	Dᵀ(z)= conv(pad_circular(z, (0,1,0,1)), Wᵀ);

    R(x) = conv(x, repeat(RESTRICT_KERNEL, 1, 1, 1, size(x, 3)); pad=1, stride=2, groups=size(x, 3))
    P(x) = begin
        C = size(x, 3)
        cdims = DenseConvDims((2size(x,1), 2size(x,2), C, 1), (3, 3, 1, C); padding=1, stride=2, groups=C) 
        4∇conv_data(x, repeat(RESTRICT_KERNEL, 1,1,1,C), cdims)
    end

    # form restricted data for restricted objective function 
    y = [y]
    for l=1:L-1
        push!(y, R(y[l]))
    end

    if isotropic
        if ℓ1
            objfun = (y, x) -> norm(y - x, 1) + λ*norm(pixelnorm(D(x)), 1)
            F₀ = (y, x, z)  -> norm(y - x, 1) + sum(z .* D(x)) + (any(pixelnorm(z) .> λ) ? Inf : 0)
            Fᵖ = (y, x, z)  -> norm(y - x, 1) + sum(z .* D(x))
            Fᵈ = (x, z)  -> sum(z .* D(x)) + (any(pixelnorm(z) .> λ) ? Inf : 0)
        else
            objfun = (y, x) -> 0.5*sum(abs2, y - x) + λ*norm(pixelnorm(D(x)), 1)
            F₀ = (y, x, z)  -> 0.5*sum(abs2, y - x) + sum(z .* D(x)) + (any(pixelnorm(z) .> λ) ? Inf : 0)
            Fᵖ = (y, x, z)  -> 0.5*sum(abs2, y - x) + sum(z .* D(x))
            Fᵈ = (x, z)  -> sum(z .* D(x)) + (any(pixelnorm(z) .> λ) ? Inf : 0)
        end
        proxgstar = z -> z ./ max.(1, pixelnorm(z)/λ)
    else
        if ℓ1
            objfun = (y, x) -> norm(y - x, 1) + λ*norm(D(x), 1)
            F₀ = (y, x, z) -> norm(x - y, 1) + sum(z .* D(x)) + (any(abs.(z) .> λ) ? Inf : 0)
            Fᵖ = (y, x, z)  -> norm(y - x, 1) + sum(z .* D(x))
            Fᵈ = (x, z)  -> sum(z .* D(x)) + (any(abs.(z) .> λ) ? Inf : 0)
        else
            objfun = (y, x) -> 0.5*sum(abs2, y - x) + λ*norm(D(x), 1)
            F₀ = (y, x, z) -> 0.5*sum(abs2, y - x) + sum(z .* D(x)) + (any(abs.(z) .> λ) ? Inf : 0)
            Fᵖ = (y, x, z)  -> 0.5*sum(abs2, y - x) + sum(z .* D(x))
            Fᵈ = (x, z)  -> sum(z .* D(x)) + (any(abs.(z) .> λ) ? Inf : 0)
        end
        proxgstar = z -> clamp.(z, -λ, λ)
    end

	if ℓ1
        proxf = (y, x) -> ST(x - y, η) + y
        ∂Fᵖ   = (y, x, z) -> sign.(x - y) + Dᵀ(z)
	else
        proxf = (y, x) -> (x + η*y) / (1 + η)  
        ∂Fᵖ   = (y, x, z) -> x - y + Dᵀ(z)
	end

    ∂i(z) = t*((z .== λ) .- (z .== -λ))
    if isotropic
        ∂Fᵈ = (x, z) -> D(x) .- ∂i(pixelnorm(z))
    else
        ∂Fᵈ = (x, z) -> D(x) .- ∂i(z)
    end

    pds_iter(y, x, z, τx, τz) = begin
        xᵏ= x
        x = proxf(y, x - η*(Dᵀ(z) - τx))
        x = x + θ*(x-xᵏ)
        z = proxgstar(z + σ*(D(x) + τz)) # should be + τz ?
        x, z
    end

    x  = similar(y); 
    z  = similar(y); 
    for l=1:L
        x[l] = zeros(T, size(y[l], 1), size(y[l], 2), 1, Channels)
        z[l] = zeros(T, size(y[l], 1), size(y[l], 2), 2, Channels)
    end

    τx = copy(x); 
    τz = copy(z); 

    u  = copy(x)
    v  = copy(z)

    w = copy(x)
    m = copy(z)

    p = copy(x)
    q = copy(z)

    k = 1
    while k ≤ maxit
        l_min = -1
        αₘ = α
        βₘ = α
        xᵏ, zᵏ = x[1], z[1]

        # first iteration starts with coarse solve
        if k > 1
            # restrict
            for l=1:L-1
                u[l], v[l] = pds_iter(y[l], x[l], z[l], τx[l], τz[l])
                for n=2:n_inner                                    
                    u[l], v[l] = pds_iter(y[l], u[l], v[l], τx[l], τz[l])
                end
                x[l+1] = R(u[l])                                         
                z[l+1] = R(v[l])                                         
                τx[l+1] = ∂Fᵖ(y[l+1], x[l+1], z[l+1]) - R(∂Fᵖ(y[l], u[l], v[l])) 
                τz[l+1] = ∂Fᵈ(x[l+1], z[l+1]) - R(∂Fᵈ(u[l], v[l]))           
            end
        end

        # solve coarsest problem exactly
        w[L], m[L] = pds_iter(y[L], x[L], z[L], τx[L], τz[L])
        for n=2:n_coarse
            w[L], m[L] = pds_iter(y[L], w[L], m[L], τx[L], τz[L])
        end

        # prolongate
        for l=L-1:-1:1
            # naive line search
            αᵏ = α
            βᵏ = α
            p[l] = u[l] + α*P(w[l+1] - x[l+1])
            q[l] = v[l] + α*P(m[l+1] - z[l+1])
            if linesearch
                bᵖ = Fᵖ(y[l], p[l], q[l]) > Fᵖ(y[l], u[l], v[l]) 
                bᵈ = Fᵈ(p[l], q[l]) < Fᵈ(u[l], v[l])
                while bᵖ || bᵈ
                    if bᵖ
                        αᵏ = αᵏ / 2
                        p[l] = u[l] + αᵏ*P(w[l+1] - x[l+1])
                    end
                    if bᵈ
                        βᵏ = βᵏ / 2
                        q[l] = v[l] + βᵏ*P(m[l+1] - z[l+1])
                    end
                    bᵖ = Fᵖ(y[l], p[l], q[l]) > Fᵖ(y[l], u[l], v[l]) 
                    bᵈ = Fᵈ(p[l], q[l]) < Fᵈ(u[l], v[l])
                    (αᵏ <= 1e-8 || βᵏ <= 1e-8) && break
                end
                l_min = αᵏ < αₘ ? l : l_min
                αₘ = min(αₘ, αᵏ)
                βₘ = min(βₘ, βᵏ)
            end

            w[l], m[l] = pds_iter(y[l], p[l], q[l], τx[l], τz[l])
            for n=2:n_inner                                    
                w[l], m[l] = pds_iter(y[l], w[l], m[l], τx[l], τz[l])
            end
        end

        # update the fine variable
        x[1], z[1] = w[1], m[1]

        O[k] = objfun(y[1], x[1])
        r[k] = norm(cat(x[1] - xᵏ, z[1] - zᵏ; dims=3)) / norm(1e-16 .+ cat(x[1], z[1]; dims=3))

        if verbose
            @printf("k: %4d | O=%.4e | r=%.4e | αₘ=%.2e | βₘ=%.2e | l_min=%d\n", k, O[k], r[k], αₘ, βₘ, l_min)
        end

        if r[k] < tol || k == maxit
            break
        end
        k += 1
    end

    x = permutedims(x[1], (1,2,4,3));
    return x[:,:,:], (k=k, obj=O[1:k], res=r[1:k])
end

