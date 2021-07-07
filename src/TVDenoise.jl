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
    tvd(y::Array{<:Real,N}, λ, ρ; maxit=100, tol=1e-2, verbose=true, itfun=(x,k)->0)

TV denoising via ADMM with sparse matrix solves (cholesky). If y is a vector,
1D TVD is performed.  If y is 2D or 3D, 2D anisotropic TV denoising is done on
the first two dimensions of y.  itfun is an optional function to be called each
iterations with the solution x and iteration number k.
"""
function tvd(y::Array{<:Real,N}, λ, ρ=1;
             isotropic  = false,
             maxit      = 100,
             tol        = 1e-2,
             verbose    = true,
	         itfun      = (x,k) -> 0) where {N}
	sz = size(y);
	D = FDmat(sz...);
	x, hist = tvd(vec(y), D, λ, ρ, isotropic, maxit, tol, verbose, itfun);
	return reshape(x, sz...), hist
end

function tvd(y::AbstractVector, D::AbstractMatrix, λ, ρ, isotropic::Bool, maxit::Int, tol, verbose::Bool, itfun::Function)
	M, N = size(D);
	x = copy(y); 
	z = zeros(M);
	u = zeros(M);
	F = zeros(maxit); # objective fun, 
	r = zeros(maxit); # primal residual
	s = zeros(maxit); # dual residual
	τ = λ/ρ

	if isotropic
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*sum(sqrt.(sum(abs2, Dx, dims=(3,4))));
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*norm(Dx, 1);
		T = ST # soft-thresholding
	end

	C = cholesky(I + ρ*D'*D);

	k = 0; itfun(x,0);
	while k == 0 || k < maxit && r[k] > tol 
		x = C\(y + ρ*D'*(z - u));
		Dx = D*x; 
		zᵏ = z;
		z = T(Dx + u, τ);
		u = u + Dx - z;       # dual ascent
		r[k+1] = norm(Dx - z);
		s[k+1] = ρ*norm(D'*(z - zᵏ));
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e | s= %.3e \n" k F[k] r[k] s[k] ;
		end
		itfun(x,k);
	end
	return x, (k=k, obj=F[1:k], pres=r[1:k], dres=s[1:k])
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
	y = reshape(y, size(y)...,1,1)
	x, hist = tvd_fft(y, args...; kw...)
	return x[:,:], hist
end

function tvd_fft(y::Array{<:Real,3}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_fft(y, args...; kw...)
	return x[:,:,:], hist
end

"""
    tvd_fft(y::Array{<:Real,N}, λ, ρ=1; maxit=100, tol=1e-2, verbose=true)

2D anisotropic TV denoising with periodic boundary conditions via ADMM. 
Accepts 2D, 3D, 4D tensors in (H,W,C,B) form, where the last two
dimensions are optional.
"""
function tvd_fft(y::Array{<:Real,4}, λ, ρ=1; isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)[1:3]
	# move channels to batch dimension
	y = permutedims(y, (1,2,4,3))
	τ = λ/ρ;

	# precompute C for x-update
	Λx = rfft([1 -1 zeros(N-2)'; zeros(M-1,N)]);
	Λy = rfft([[1; -1; zeros(M-2)] zeros(M,N-1)])
	C = 1 ./ ( 1 .+ ρ.*(abs2.(Λx) .+ abs2.(Λy)) );

	# real Fourier xfrm in image dimension.
	# Must specify length of first dimension for inverse.
	Q  = plan_rfft(y,(1,2)); 
	Qᴴ = plan_irfft(rfft(y),M,(1,2));

	# conditional function definition requires anonymous functions 
	if isotropic
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*sum(sqrt.(sum(abs2, Dx, dims=(3,4))));
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*norm(Dx, 1);
		T = ST # soft-thresholding
	end

	# initialization
	x = zeros(M,N,1,P);
	Dxᵏ = zeros(M,N,2,P);
	z = zeros(M,N,2,P);
	u = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store primal residual
	s = zeros(maxit); # store dual   residual

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
	while k == 0 || k < maxit && r[k] > tol && s[k] > tol
		x = Qᴴ*(C.*(Q*( y + ρ*Dᵀ!(x,z-u) ))); # x update
		Dxᵏ = D!(Dxᵏ,x);
		zᵏ  = z;
		z = T(Dxᵏ+u, τ);                   # z update
		u = u + Dxᵏ - z;                   # dual ascent
		r[k+1] = norm(Dxᵏ-z);
		s[k+1] = norm(z-zᵏ);
		F[k+1] = objfun(x,Dxᵏ);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e | s= %.3e \n" k F[k] r[k] s[k];
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x, (k=k, obj=F[1:k], pres=r[1:k], dres=s[1:k])
end

"""
    tvd_fft(y, args...; kw...)

tvd_fft with ability to pass in image types (ex. typeof(y) = Matrix{RGB{N0f8},2}).
"""
function tvd_pds(y, args...; kw...)
	xt, hist = tvd_pds(img2tensor(y), args...; kw...)
	return tensor2img(xt), hist
end

function tvd_pds(y::Array{<:Real,2}, args...; kw...) 
	y = reshape(y, size(y)...,1,1)
	x, hist = tvd_pds(y, args...; kw...)
	return x[:,:], hist
end

function tvd_pds(y::Array{<:Real,3}, args...; kw...) 
	y = reshape(y, size(y)...,1)
	x, hist = tvd_pds(y, args...; kw...)
	return x[:,:,:], hist
end

function tvd_pds(y::Array{<:Real,4}, λ, γ1, γ2; isotropic=false, maxit=100, tol=1e-2, verbose=true) 
	M, N, P = size(y)[1:3]
	# move channels to batch dimension
	y = permutedims(y, (1,2,4,3))

	# conditional function definition requires anonymous functions 
	if isotropic
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*sum(sqrt.(sum(abs2, Dx, dims=(3,4))));
		T = BT # block-thresholding
	else
		objfun = (x,Dx) -> 0.5*sum(abs2.(x-y)) + λ*norm(Dx, 1);
		T = ST # soft-thresholding
	end

	# initialization
	x = zeros(M,N,1,P);
	z = zeros(M,N,2,P);
	u = zeros(M,N,2,P);
	F = zeros(maxit); # store objective fun
	r = zeros(maxit); # store primal residual

	# conv kernel
	W = zeros(Float64, 2,2,1,2);
	W[:,:,1,1] = [1 -1; 0 0];          # dx
	W[:,:,1,2] = [1  0;-1 0];          # dy
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	
	D(x) = conv(pad_circular(x, (1,0,1,0)), W);
	Dᵀ(z)= conv(pad_circular(z, (0,1,0,1)), Wᵀ);

	Y = norm(y)

	k = 0;
	while k == 0 || k < maxit && r[k] > tol 
		xᵏ= x
		x = 0.5*(x - γ1*Dᵀ(z) + y)
		u = z + γ2*D(2x - xᵏ)
		z = u - T(u,λ)
		Dx = D(x)
		r[k+1] = norm(x - xᵏ)/Y;
		F[k+1] = objfun(x,Dx);
		k += 1;
		if verbose
			@printf "k: %3d | F= %.3e | r= %.3e \n" k F[k] r[k] ;
		end
	end
	x = permutedims(x, (1,2,4,3));
	return x, (k=k, obj=F[1:k], pres=r[1:k])
end

end # module

