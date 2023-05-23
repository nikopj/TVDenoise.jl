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

function mg_tvd_pds(y::Array{T,3}, λ, L=1; isotropic=true, ℓ1=false, α=0.1, θ=0, maxit=100, n_inner=1, n_coarse=100, tol=1e-5, verbose=true, linesearch=true) where T
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
    ∂Fᵈ(x, z) = D(x) 

    pds_iter(y, x, z, τx, τz) = begin
        xᵏ= x
        x = proxf(y, x - η*(Dᵀ(z) - τx))
        x = x + θ*(x-xᵏ)
        z = proxgstar(z + σ*(D(x) + τz))
        x, z
    end

    x  = similar(y); x[1] = zeros(M, N, 1, Channels)
    z  = similar(y); z[1] = zeros(M, N, 2, Channels)
    τx = similar(y); τx[1] = zero(x[1])
    τz = similar(y); τz[1] = zero(z[1])

    u  = similar(y)
    v  = similar(y)

    w  = similar(y); w[L] = zeros(size(y[L], 1), size(y[L], 2), 1, Channels)
    m  = similar(y); m[L] = zeros(size(y[L], 1), size(y[L], 2), 2, Channels)

    p  = similar(y)
    q  = similar(y)

    k=1
    for k=1:maxit
        αₘ = α
        xᵏ, zᵏ = x[1], z[1]

        # restrict
        for l=1:L-1
            u[l], v[l] = pds_iter(y[l], x[l], z[l], τx[l], τz[l])
            for n=2:n_inner                                    
                u[l], v[l] = pds_iter(y[l], u[l], v[l], τx[l], τz[l])
            end
            x[l+1] = R(u[l])                                         
            z[l+1] = R(v[l])                                         
            τx[l+1] = ∂Fᵖ(y[l+1], x[l+1], z[l+1]) - R(∂Fᵖ(y[l], u[l], v[l])) 
            τz[l+1] = ∂Fᵈ(x[l+1], z[l+1]) - R(∂Fᵈ(x[l], z[l]))           
        end

        # solve coarsest problem exactly
        for n=1:n_coarse
            w[L], m[L] = pds_iter(y[L], w[L], m[L], τx[L], τz[L])
        end

        # prolongate
        for l=L-1:-1:1
            # naive line search
            αᵏ = α
            p[l] = u[l] + α*P(w[l+1] - x[l+1])
            q[l] = v[l] + α*P(m[l+1] - z[l+1])
            if linesearch
                while Fᵖ(y[l], p[l], q[l]) > Fᵖ(y[l], u[l], v[l])
                    αᵏ = αᵏ / 2
                    αᵏ < 1e-8 && break
                    p[l] = u[l] + αᵏ*P(w[l+1] - x[l+1])

                    if Fᵈ(p[l], q[l]) < Fᵈ(u[l], v[l])
                        q[l] = v[l] + αᵏ*P(m[l+1] - z[l+1])
                    end
                end
                αₘ = min(αₘ, αᵏ)
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
            @printf("k: %4d | O=%.5e | r=%.5e | αₘ=%.5e\n", k, O[k], r[k], αₘ)
        end

        if r[k] < tol
            break
        end
    end

    x = permutedims(x[1], (1,2,4,3));
    return x[:,:,:], (k=k, obj=O[1:k], res=r[1:k])
end

