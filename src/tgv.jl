function fd2kernel(T::Type)
	W = zeros(T, 2,2,2,4)
	W[:,:,1,1] =     [1 -1; 0 0]; 
	W[:,:,2,1] =     [0  0; 0 0]; 

	W[:,:,1,2] =     [1  0;-1 0]; 
	W[:,:,2,2] =     [0  0; 0 0]; 

	W[:,:,1,3] =     [0  0; 0 0]; 
	W[:,:,2,3] =     [1 -1; 0 0]; 

	W[:,:,1,4] =     [0  0; 0 0]; 
	W[:,:,2,4] =     [1  0;-1 0]; 
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

"""
    tgv_pds(y, λ; kw...)

Inputs may be images, matries, or tensors.
"""
function tgv_pds(y, args...; kw...)
	x, hist = tgv_pds(img2tensor(y), args...; kw...)
	return tensor2img(x), hist
end

function tgv_pds(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tgv_pds(y, args...; kw...)
	return x[:,:], hist
end

"""
    tgv_pds(y::Array{<:Real,3}, λ; ℓ1=false, θ=0, isotropic=false, maxit=100, tol=1e-2, verbose=true)

TV Denoising via Primal-Dual splitting, also known as Chambolle-Pock Algorithm,
Primal Dual Hybrid Gradient, or (in this case, θ=0) the Arrow Hurwitz Uzawa
Algorithm. Lagrange multiplier λ required. Step-sizes set to maximum
internally. Keyword argument ℓ1 minimizes an ℓ1 data-fidelity term instead of ℓ2.
"""
function tgv_pds(y::Array{<:Real,3}, λ₀, λ₁; ℓ1=false, θ=0, isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y, size(y)...,1)  # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension

	# set step-sizes at maximum: τσL² < 1
	# note: PDS seems sensitive to these (given finite iterations at least...)
	L = sqrt(12) # Spectral norm of D
    τ = 0.99 / L
    σ = 0.99 / L

	if ℓ1
        proxf = (x) -> ST(x - y, τ) + y
	else
        proxf = (x) -> (x + τ*y)/(1+τ)  
	end

	if isotropic
		if ℓ1
			objfun = (x,z,Dx,Dz) -> norm(x-y,1) + λ₀*norm(pixelnorm(Dx-z),1) + λ₁*norm(pixelnorm(Dz),1)
		else
			objfun = (x,z,Dx,Dz) -> 0.5*sum(abs2, x-y) + λ₀*norm(pixelnorm(Dx-z),1) + λ₁*norm(pixelnorm(Dz),1)
		end
		# pixel-vector projection onto λ-2-norm ball
		Π = (z, λ) -> z ./ max.(1,pixelnorm(z)/λ)
	else
		if ℓ1
			objfun = (x,z,Dx,Dz) -> norm(x-y,1) + λ₀*norm(Dx-z,1) + λ₁*norm(Dz, 1)
		else
			objfun = (x,z,Dx,Dz) -> 0.5*sum(abs2, x-y) + λ₀*norm(Dx-z, 1) + λ₁*norm(Dz, 1)
		end
		# coeff projection onto λ-inf-norm ball
		Π = (z, λ) -> clamp.(z, -λ, λ) 
	end

	# initialization
    x  = zeros(M,N,1,P);
	z  = zeros(M,N,2,P);
    Dx = zeros(M,N,2,P);
	u  = zeros(M,N,2,P);
	v  = zeros(M,N,4,P);
	Dz = zeros(M,N,4,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store primal residual

	# conv kernel
	W, Wᵀ = fdkernel(eltype(y))
	D(x)  = conv(pad_circular(x, (1,0,1,0)), W);
	Dᵀ(u) = conv(pad_circular(u, (0,1,0,1)), Wᵀ);

	# (in-place) Circular convolution
	cdims = DenseConvDims(pad_circular(x, (1,0,1,0)), W);
	cdimsᵀ= DenseConvDims(pad_circular(u, (0,1,0,1)), Wᵀ);
	D!(u,x) = conv!(u, pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ!(x,u)= conv!(x, pad_circular(u, (0,1,0,1)), Wᵀ,cdimsᵀ);

	W2, W2ᵀ = fd2kernel(eltype(y))
    D2(z)  = conv(pad_circular(z, (1,0,1,0)), W2);
    D2ᵀ(v) = conv(pad_circular(v, (0,1,0,1)), W2ᵀ);

	# (in-place) Circular convolution
	cdims2 = DenseConvDims(pad_circular(z, (1,0,1,0)), W2);
	cdims2ᵀ= DenseConvDims(pad_circular(v, (0,1,0,1)), W2ᵀ);
	D2!(v,z) = conv!(v, pad_circular(z, (1,0,1,0)), W2, cdims2);
	D2ᵀ!(z,v)= conv!(z, pad_circular(v, (0,1,0,1)), W2ᵀ,cdims2ᵀ);

	k = 0;
	while k < 2 || k < maxit && r[k] > tol 
		xᵏ= x 
		zᵏ= z 
		x = proxf(x - τ*Dᵀ(u))           # Proximal Gradient Descent on primals (x, z)
        z = z - τ*(D2ᵀ(v) - u)
		x = x + θ*(x-xᵏ);                # Extrapolation (not necessary for smooth data term)
		z = z + θ*(z-zᵏ);             
	    D!(Dx, x)
		D2!(Dz, z)
		u = Π(u + σ*(Dx - z), λ₀)        # Proximal Gradient Ascent on duals (u, v)
		v = Π(v + σ*Dz, λ₁)             
        r[k+1] = norm(cat(x - xᵏ, z - zᵏ; dims=3))/(norm(cat(x, z; dims=3)) + 1e-16);
		F[k+1] = objfun(x, z, Dx, Dz);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k] 
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end
