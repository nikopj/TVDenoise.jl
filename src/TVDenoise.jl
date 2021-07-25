module TVDenoise
#=
Implementation of Total Variation denoising via two ADMM based methods: sparse
matrix direct solves and FFT solves 
=#
include("utils.jl")
export tvd, tvd_fft, tvd_pds, img2tensor, tensor2img
using Printf, LinearAlgebra, FFTW, NNlib

"""
    tvd(y, args...; kw...)

tvd with ability to pass in image types (ex. typeof(y) = Matrix{RGB{N0f8},2}).
"""
function tvd(y, args...; kw...)
	xt, hist = tvd(img2tensor(y), args...; kw...)
	return tensor2img(xt), hist
end

"""
    tvd(y::Array{<:Real,N}, λ, ρ; maxit=100, tol=1e-2, verbose=true)

TV denoising via ADMM with sparse matrix solves (cholesky). If y is a vector,
1D TVD is performed.  If y is 2D or 3D, 2D anisotropic TV denoising is done on
the first two dimensions of y.  
"""
function tvd(y::Array{<:Real,N}, λ, ρ=1;
             isotropic  = false,
             maxit      = 100,
             tol        = 1e-2,
             verbose    = true) where {N}
	sz = size(y);
	D = FDmat(sz...);
	x, hist = tvd(vec(y), D, λ, ρ, isotropic, maxit, tol, verbose);
	return reshape(x, sz...), hist
end

function tvd(y::AbstractVector, D::AbstractMatrix, λ, ρ, isotropic::Bool, maxit::Int, tol, verbose::Bool)
	M, N = size(D);
	x = copy(y); 
	z = zeros(M);
	u = zeros(M);
	F = zeros(maxit); # objective fun, 
	r = zeros(maxit); # store normalized residual
	τ = λ/ρ

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		T = ST # soft-thresholding
	end

	C = cholesky(I + ρ*D'*D);

	k = 0
	while k == 0 || k < maxit && r[k] > tol 
		xᵏ= x
		x = C\(y + ρ*D'*(z - u));      # x-update
		Dx = D*x; 
		zᵏ = z;
		z = T(Dx + u, τ);              # z-update
		u = u + Dx - z;                # dual ascent
		r[k+1] = norm(x - xᵏ)/norm(x);
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k]
		end
	end
	return x, (k=k, obj=F[1:k], res=r[1:k])
end

"""
    tvd_fft(y, args...; kw...)

tvd_fft with ability to pass in image types (ex. typeof(y) = Matrix{RGB{N0f8},2}).
"""
function tvd_fft(y, args...; kw...)
	xt, hist = tvd_fft(img2tensor(y), args...; kw...)
	return tensor2img(xt), hist
end

function tvd_fft(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_fft(y, args...; kw...)
	return x[:,:], hist
end

"""
    tvd_fft(y::Array{<:Real,N}, λ, ρ=1; maxit=100, tol=1e-2, verbose=true)

2D anisotropic TV denoising with periodic boundary conditions via ADMM. 
Accepts 2D, 3D, 4D tensors in (H,W,C,B) form, where the last two
dimensions are optional.
"""
function tvd_fft(y::Array{<:Real,3}, λ, ρ=1; isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y,size(y)...,1)   # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ/ρ;

	# precompute C for x-update
	Λx = rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	C = 1 ./ ( 1 .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) );

	# real Fourier xfrm in image dimension.
	# Must specify length of first dimension for inverse.
	Q  = plan_rfft(y,(1,2)); 
	Qᴴ = plan_irfft(rfft(y),M,(1,2));

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		T = ST # soft-thresholding
	end

	# initialization
	x = zeros(M,N,1,P);
	Dxᵏ = zeros(M,N,2,P);
	z = zeros(M,N,2,P);
	u = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store (relative) primal residual

	# conv kernel
	W = zeros(Float64, 2,2,1,2);
	W[:,:,1,1] = [1 -1; 0 0];          # dx
	W[:,:,1,2] = [1  0;-1 0];          # dy
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);

	# (in-place) Circular convolution
	cdims = DenseConvDims(pad_circular(x, (1,0,1,0)), W);
	cdimsᵀ= DenseConvDims(pad_circular(z, (0,1,0,1)), Wᵀ);
	D!(z,x) = conv!(z, pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ!(x,z)= conv!(x, pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);

	k = 0;
	while k == 0 || k < maxit && r[k] > tol
		xᵏ= x
		x = Qᴴ*(C.*(Q*( y + ρ*Dᵀ!(x,z-u) ))); # x update
		Dxᵏ = D!(Dxᵏ,x);
		zᵏ  = z;
		z = T(Dxᵏ+u, τ);              # z update
		u = u + Dxᵏ - z;              # dual ascent
		r[k+1] = norm(x - xᵏ)/norm(x);
		F[k+1] = objfun(x,Dxᵏ);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k];
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end

"""
    tvd_pds(y, λ, γ1, γ2; kw...)

TV Denoising via Primal-Dual splitting, also known as Chambolle-Pock Algorithm,
Primal Dual Hybrid Gradient, or (in this case, θ=0) the Arrow Hurwitz Uzawa
Algorithm. Lagrange multiplier λ required. Step-sizes set to maximum
internally.
"""
function tvd_pds(y, args...; kw...)
	xt, hist = tvd_pds(img2tensor(y), args...; kw...)
	return tensor2img(xt), hist
end

function tvd_pds(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_pds(y, args...; kw...)
	return x[:,:], hist
end

function tvd_pds(y::Array{<:Real,3}, λ; isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y, size(y)...,1)  # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		# pixel-vector projection onto λ-2-norm ball
		Π = z -> z ./ max.(1,pixelnorm(z)/λ)
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		# coeff projection onto λ-inf-norm ball
		Π = z -> clamp.(z, -λ, λ) 
	end

	# set step-sizes at maximum: τσL² < 1
	# note: PDS seems sensitive to these (given finite iterations at least...)
	L = sqrt(8) # Spectral norm of D
	τ = 0.99 / L
	σ = 0.99 / L

	# initialization
	x = zeros(M,N,1,P);
	z = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	G = zeros(maxit); # store primal-dual gap
	r = zeros(maxit); # store primal residual

	# conv kernel
	W = zeros(Float64, 2,2,1,2);
	W[:,:,1,1] = [1 -1; 0 0];          # dx
	W[:,:,1,2] = [1  0;-1 0];          # dy
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	
	D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
	Dᵀ(z)= conv(pad_constant(z, (0,1,0,1), dims=(1,2)), Wᵀ);

	k = 0;
	while k == 0 || k < maxit && r[k] > tol 
		xᵏ= x; x = x - τ*(x - y + Dᵀ(z)) # Gradient Descent on x (primal)
		#x = 2x - xᵏ;                    # Extrapolation (not necessary for smooth data term)
		Dx = D(x)
		z = Π(z + σ*Dx)                  # Proximal Gradient Ascent on z (dual)
		r[k+1] = norm(x - xᵏ)/norm(x);
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k]
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end

function flow(I::Array{<:Real,4}, λ; maxit=100, tol=1e-2, verbose=true) 
	M, N, C, B = size(I)
	@assert B==2 "Must provide two input images!"

	# conv kernel
	W = zeros(Float64, 2,2,1,2);
	W[:,:,1,1] = [1 -1; 0 0];          # dx
	W[:,:,1,2] = [1  0;-1 0];          # dy
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	
	D = (r,v) -> conv(pad_constant(v, (1,0,1,0), dims=(1,2)), repeat(W,  outer=(1,1,r,r)))
	Dᵀ= (r,w) -> conv(pad_constant(w, (0,1,0,1), dims=(1,2)), repeat(Wᵀ, outer=(1,1,r,r)))
	
	I = permutedims(I, (1,2,4,3)) # swap channels and batch dimension
	Iₜ = I[:,:,2] - I[:,:,1]      # time derivative
	∇I = D(1,I[:,:,1:1,:])          # space derivative

	pixeldot = (a,b) -> sum(a.*b, dims=(3,4))
	objfun = (v,Dv) -> 0.5*sum(pixeldot(∇I,v)+Iₜ) + λ*norm(pixelnorm(Dv),1)

	α = pixelnorm(∇I).^2
	proxf = (v,ρ) -> v + ∇I.*( ST(ρ, τ*α) - ρ )./α
	Π = w -> w ./ max.(1,pixelnorm(w)/λ)

	# set step-sizes at maximum: τσL² < 1
	# note: PDS seems sensitive to these (given finite iterations at least...)
	L = sqrt(8) # Spectral norm of D
	τ = 0.99 / L
	σ = 0.99 / L

	# initialization
	v = zeros(M,N,2,C);
	w = zeros(M,N,4,C);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store primal residual

	k = 0;
	while k == 0 || k < maxit && r[k] > tol 
		vᵏ= v; ρ = pixeldot(∇I,v) .+ Iₜ
		v = proxf(v - τ*Dᵀ(2,w), ρ)                 # Proximal Gradient Descent on x (primal)
		v = 2v - vᵏ;                    # Extrapolation 
		Dv = D(2,v)
		w = Π(w + σ*Dv)                 # Proximal Gradient Ascent on z (dual)
		r[k+1] = norm(v - vᵏ)/norm(v);
		F[k+1] = objfun(v,Dv);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k]
		end
	end
	v = permutedims(v, (1,2,4,3));
	return v[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end

end # module

