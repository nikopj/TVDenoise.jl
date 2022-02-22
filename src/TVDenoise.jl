module TVDenoise
#=
Implementation of Total Variation denoising via two ADMM based methods: sparse
matrix direct solves and FFT solves 
=#
include("utils.jl")
export tvd, tvd_fft, tvd_pds, img2tensor, tensor2img, flow, saltpepper!, saltpepper, bayer_mask, tvd_vamp
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
	D = fdmat(sz...);
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
function tvd_fft(y::Array{<:Real,3}, λ, ρ=1; h=missing, isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y,size(y)...,1)   # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension
	τ = λ/ρ;

	if !ismissing(h)
		h′ = [h zeros(size(h,1), N-size(h,2)); 
		      zeros(M-size(h,1), size(h,2)) zeros((M,N) .- size(h))]
		@show size(h′)
		Σ = rfft(h′)
	else
		Σ = 1
	end

	# precompute C for x-update
	Λx = rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	C = 1 ./ ( abs2.(Σ) .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) )

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
	W, Wᵀ = fdkernel(eltype(y))

	# (in-place) Circular convolution
	cdims = DenseConvDims(pad_circular(x, (1,0,1,0)), W);
	cdimsᵀ= DenseConvDims(pad_circular(z, (0,1,0,1)), Wᵀ);
	D!(z,x) = conv!(z, pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ!(x,z)= conv!(x, pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);
	D(x) = conv(pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ(z)= conv(pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);

	if !ismissing(h)
		h = h[:,:,:,:]
		hᵀ= rot180(h[:,:])[:,:,:,:]
		padu, padd = ceil(Int,(size(h,1)-1)/2), floor(Int,(size(h,1)-1)/2)
		padl, padr = ceil(Int,(size(h,2)-1)/2), floor(Int,(size(h,2)-1)/2)
		pad1 = (padu, padd, padl, padr)
		pad2 = (padd, padu, padr, padl)
		# cdims reference being kept, rename variable to cdims2
		cdims2 = DenseConvDims(pad_circular(x, pad1), h);
		cdims2ᵀ= DenseConvDims(pad_circular(x, pad2), hᵀ);
		H = x->conv(pad_circular(x, pad1), h, cdims2);
		Hᵀ= x->conv(pad_circular(x, pad2), hᵀ,cdims2ᵀ);
	else
		H = identity
		Hᵀ= identity
	end

	@show Dᵀ!(x,z) |> size

	k = 0;
	while k == 0 || k < maxit && r[k] > tol
		xᵏ= x 
		x = Qᴴ*(C.*(Q*( Hᵀ(y) + ρ*Dᵀ(z-u) ))); # x update
		D!(Dxᵏ,x);
		zᵏ  = z;
		z = T(Dxᵏ+u, τ);                      # z update
		u = u + Dxᵏ - z;                      # dual ascent
		r[k+1] = norm(x - xᵏ)/norm(x);
		F[k+1] = objfun(H(x),Dxᵏ);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k];
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end

"""
    tvd_pds(y, λ; kw...)

Inputs may be images, matries, or tensors.
"""
function tvd_pds(y, args...; kw...)
	x, hist = tvd_pds(img2tensor(y), args...; kw...)
	return tensor2img(x), hist
end

function tvd_pds(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_pds(y, args...; kw...)
	return x[:,:], hist
end

"""
    tvd_pds(y::Array{<:Real,3}, λ; ℓ1=false, θ=0, isotropic=false, maxit=100, tol=1e-2, verbose=true)

TV Denoising via Primal-Dual splitting, also known as Chambolle-Pock Algorithm,
Primal Dual Hybrid Gradient, or (in this case, θ=0) the Arrow Hurwitz Uzawa
Algorithm. Lagrange multiplier λ required. Step-sizes set to maximum
internally. Keyword argument "ℓ1" minimizes an ℓ1 data-fidelity term instead of ℓ2.
"""
function tvd_pds(y::Array{<:Real,3}, λ; ℓ1=false, θ=0, isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y, size(y)...,1)  # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension

	if ℓ1
		proxf = x -> ST(x - y, τ) + y
	else
		proxf = x -> (x + τ*y)/(1+τ)  
	end

	if isotropic
		if ℓ1
			objfun = (x,Dx) -> norm(x-y,1) + λ*norm(pixelnorm(Dx),1)
		else
			objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		end
		# pixel-vector projection onto λ-2-norm ball
		Π = z -> z ./ max.(1,pixelnorm(z)/λ)
	else
		if ℓ1
			objfun = (x,Dx) -> norm(x-y,1) + λ*norm(Dx,1)
		else
			objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		end
		# coeff projection onto λ-inf-norm ball
		Π = z -> clamp.(z, -λ, λ) 
	end

	# set step-sizes at maximum: τσL² < 1
	# note: PDS seems sensitive to these (given finite iterations at least...)
	L = sqrt(8) # Spectral norm of D
	τ = 0.99 / L
	σ = 0.99 / L

	# initialization
	x  = zeros(M,N,1,P);
	z  = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store primal residual

	# conv kernel
	W, Wᵀ = fdkernel(eltype(y))
	D(x) = conv(pad_constant(x, (1,0,1,0), dims=(1,2)), W);
	Dᵀ(z)= conv(pad_constant(z, (0,1,0,1), dims=(1,2)), Wᵀ);

	k = 0;
	while k == 0 || k < maxit && r[k] > tol 
		xᵏ= copy(x); 
		x = proxf(x - τ*Dᵀ(z))           # Proximal Gradient Descent on x (primal)
		x = x + θ*(x-xᵏ);                # Extrapolation (not necessary for smooth data term)
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

"""
    tvd_vamp(y, args...; kw...)

tvd_vamp with ability to pass in image types (ex. typeof(y) = Matrix{RGB{N0f8},2}).
"""
function tvd_vamp(y, args...; kw...)
	xt, hist = tvd_vamp(img2tensor(y), args...; kw...)
	return tensor2img(xt), hist
end

function tvd_vamp(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_vamp(y, args...; kw...)
	return x[:,:], hist
end

"""
    tvd_vamp(y::Array{<:Real,N}, λ; γ=0.6, maxit=100, tol=1e-2, verbose=true)

2D anisotropic TV denoising with periodic boundary conditions via VAMP,
Vector Approximate Message Passing.
Over-relaxation parameter γ helps with monotonic convergence.
Accepts 2D, 3D, 4D tensors in (H,W,C,B) form, where the last two
dimensions are optional.
"""

function tvd_vamp(y::Array{<:Real,3}, λ; isotropic=false, γ=0.6, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)
	y = reshape(y,size(y)...,1)   # unsqueeze for NNlib conv
	y = permutedims(y, (1,2,4,3)) # move channels to batch dimension

	# precompute C for x-update
	Λx = rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	B = abs2.(Λx) .+ abs2.(Λy)
	C(ρ) = 1 ./ ( 1 .+ ρ.*B );

	# real Fourier xfrm in image dimension.
	# Must specify length of first dimension for inverse.
	Q  = plan_rfft(y,(1,2)); 
	Qᴴ = plan_irfft(rfft(y),M,(1,2));

	if isotropic
		objfun = (x,Dx) -> objfun_iso(x,Dx,y,λ)
		T = BT # block-thresholding
		divT = (x,τ) -> begin 
			normx = sqrt.(sum(abs2, x, dims=(3,4)))
			loc = normx .> τ
			sum(loc .* ( 1 .- τ*( 1 .- (x./normx).^2)./normx))
		end
	else
		objfun = (x,Dx) -> objfun_aniso(x,Dx,y,λ)
		T = ST # soft-thresholding
		divT = (x,τ) -> sum(abs.(x) .> τ)
	end

	# initialization
	ρ = 1
	x = zeros(M,N,1,P);
	v = zeros(M,N,1,P);
	Dx= zeros(M,N,2,P);
	z = zeros(M,N,2,P);
	u = zeros(M,N,2,P);
	w = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store (relative) primal residual

	# conv kernel
	W, Wᵀ = fdkernel(eltype(y))

	# (in-place) Circular convolution
	cdims = DenseConvDims(pad_circular(x, (1,0,1,0)), W);
	cdimsᵀ= DenseConvDims(pad_circular(z, (0,1,0,1)), Wᵀ);
	D!(z,x) = conv!(z, pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ!(x,z)= conv!(x, pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);
	D(x) = conv(pad_circular(x, (1,0,1,0)), W, cdims);
	Dᵀ(z)= conv(pad_circular(z, (0,1,0,1)), Wᵀ,cdimsᵀ);

	k = 0;
	while k == 0 || k < maxit && r[k] > tol
		x′ = x
		C′= C(ρ)                               # Fourier factor
		x = Qᴴ*(C′.*(Q*( y + Dᵀ!(v,u) )));        # x update

		# below: factor of P in numerator and denominator cancel
		σx= sum(B.*C′) / (M*N)
		τ = λ*σx / (1 - σx*ρ)
		Dx= D!(Dx,x);

		w = (Dx - σx*u)./(1-σx*ρ)
		z = T(w, τ)        # z update
		σz= τ * divT(w,τ) / (λ*2*M*N*P)

		u = u + γ*(z/σz - Dx/σx)             # dual ascent
		ρ = ρ + γ*((1/σz) - (1/σx))

		r[k+1] = norm(x - x′)/norm(x);
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e | ρ= %.3e | σx= %.3e | σz= %.3e \n" k F[k] r[k] ρ σx σz;
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x[:,:,:], (k=k, obj=F[1:k], res=r[1:k])
end

end # module

