using Images, NNlib, SparseArrays, Random

HT(x,τ) = x*(abs(x) > τ);                      # hard-thresholding
ST(x,τ) = sign.(x).*max.(abs.(x).-τ, 0);       # soft-thresholding
pixelnorm(x) = sqrt.(sum(abs2, x, dims=(3,4))) # 2-norm on 4D image-tensor pixel-vectors
BT(x,τ) = max.(1 .- τ ./ pixelnorm(x), 0).*x   # block-thresholding

objfun_iso(x,Dx,y,λ)   = 0.5*sum(abs2.(x-y)) + λ*norm(pixelnorm(Dx), 1) 
objfun_aniso(x,Dx,y,λ) = 0.5*sum(abs2.(x-y)) + λ*norm(Dx, 1)

function tensor2img(A::Array{<:Real,2})
	tensor2img(Gray, A)
end

function tensor2img(A::Array{<:Real,3})
	tensor2img(RGB, permutedims(A, (3,1,2)))
end

function tensor2img(ctype, A::Array{<:Real,N}) where {N}
	reinterpret(reshape, ctype{N0f8}, N0f8.(clamp.(A,0,1)))
end

function img2tensor(A)
	B = Float64.(reinterpret(reshape, N0f8, A) |> collect)
	if ndims(B) == 3
		B = permutedims(B, (2,3,1))
	end
	return B
end

"""
    fdmat(M,[N,[C]])::SparseMatrixCSC

Return First order Derivative matrix for 1D/2D/3D matrix.
2D is anisotropic. 3D is concatenation of 2D in channel dimension.
"""
function fdmat(N::Int)::SparseMatrixCSC
	spdiagm(0=>-1*ones(N), 1=>ones(N))[1:N-1,1:N]; 
end

function fdmat(M::Int, N::Int)::SparseMatrixCSC
	# vertical derivative
	S = spdiagm(N-1, N, ones(N-1));
	T = fdmat(M);
	Dy = kron(S,T);
	# horizontal derivative
	S = fdmat(N);
	T = spdiagm(M-1, M, ones(M-1));
	Dx = kron(S,T);
	return [Dx; Dy];
end

function fdmat(M::Int, N::Int, C::Int)::SparseMatrixCSC
	kron(I(C),fdmat(M,N))
end

"""
    pad_circular(A::Array{<:Number,4}, pad::NTuple{Int,4})

Pads A in first two dimensions circularly. Pad gives padding in top, 
bottom, left, right order.

Example:
```julia
julia> A = reshape(1:9, 3,3,1,1)
3×3×1×1 reshape(::UnitRange{Int64}, 3, 3, 1, 1) with eltype Int64:
[:, :, 1, 1] =
 1  4  7
 2  5  8
 3  6  9

julia> B = pad_circular(A, (1,0,1,0))
4×4×1×1 Array{Int64, 4}:
[:, :, 1, 1] =
 9  3  6  9
 7  1  4  7
 8  2  5  8
 9  3  6  9

```
"""
function pad_circular(A::Array{<:Number,4}, pad::NTuple{4,Int})
	# allocate array
	M, N = size(A)[1:2]
	if any(pad .> M) || any(pad .> N)
		error("padding larger than original matrix!")
	end
	B = pad_constant(A, pad, dims=(1,2))
	t, b, l, r = pad
	f(p,L) = L-p+1:L
	# top-left lorner 
	B[1:t, 1:l, :, :]               = A[f(t,M), f(l,N), :, :]
	# top-middle 
	B[1:t, l+1:l+N, :, :]           = A[f(t,M), :, :, :]
	# top-right
	B[1:t, f(r,N+l+r), :, :]        = A[f(t,M), 1:r, :, :]
	# left-middle
	B[t+1:t+M, 1:l, :, :]           = A[:, f(l,N), :, :]
	# right-middle
	B[t+1:t+M, f(r,N+l+r), :, :]    = A[:, 1:r, :, :]
	# bottom-left
	B[f(b,M+b+t), 1:l, :, :]        = A[1:b, f(l,N), :, :]
	# bottom-middle
	B[f(b,M+t+b), l+1:l+N, :, :]    = A[1:b, :, :, :]
	# bottom-right
	B[f(b,M+t+b), f(r,N+l+r), :, :] = A[1:b, 1:r, :, :]
	return B
end

"""
    fdkernel(T::Type)

Return 2D forward difference convolution kernels (analysis and synthesis) for
use with NNlib's conv.
"""
function fdkernel(T::Type)
	W = zeros(T, 2,2,1,2)
	W[:,:,1,1] = [1 -1; 0 0]; 
	W[:,:,1,2] = [1  0;-1 0]; 
	Wᵀ = reverse(permutedims(W, (2,1,4,3)), dims=:);
	return W, Wᵀ
end

"""
    saltpepper!(x, p=0.5)

Add salt and pepper noise to array x with probability p.  Salt and pepper come
in equal proportions.
"""
function saltpepper!(x::Array{T,M}, p=0.5) where {T<:Real,M}
	N = length(x)
	ω = rand(eltype(x), N) .< p
	Ω = shuffle(1:N)[ω]
	m = length(Ω)
	Ωₛ, Ωₚ = Ω[1:m÷2], Ω[m÷2+1:end]
	x[Ωₛ] .= one(T)
	x[Ωₚ] .= zero(T)
	return x
end
saltpepper(x, p) = saltpepper!(copy(x), p)

"""
    bayer_mask(T::Type, M::Int, N::Int)

Return the RGGB Bayer Color-Filter-Array of type T and size MxN.
Result is an Array{T,3} where size(w) = (M,N,3).
"""
function bayer_mask(T::Type,M::Int,N::Int)
	w = zeros(T,M,N,3)
	w[1:2:end, 1:2:end, 1] .= 1 # R
	w[2:2:end, 1:2:end, 2] .= 1 # G1
	w[1:2:end, 2:2:end, 2] .= 1 # G2
	w[2:2:end, 2:2:end, 3] .= 1 # B
	return w
end
bayer_mask(x::AbstractArray) = bayer_mask(eltype(x), size(x)[1:2]...)

function gaussiankernel(T::Type, σ::Real, m::Int=ceil(Int,6σ-1))
	K = zeros(T, m, m, 1, 1)
	c = (m-1)/2
	for i=0:m-1, j=0:m-1
		K[i+1,j+1,1,1] = exp(-((i-c)^2 + (j-c)^2) / (2σ^2))
	end
	return K./sum(K)
end
gaussiankernel(σ, m=ceil(Int,6σ-1)) = gaussiankernel(Float32, σ, m)

