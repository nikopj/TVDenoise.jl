using TVDenoise
using Plots, Printf, TestImages, FileIO
#=
Example TV denoising!
=#

# generate noisy image
#img = testimage("fabio_color_256")
img = load("cameraman.jpg")
I = img2tensor(img)
y = I + 0.1*randn(size(I))
@info size(y)

# TVD parameters
λ = 0.12; ρ = 2
kw = Dict(:isotropic=>false, :maxit=>500, :tol=>1e-4, :verbose=>false)
@info kw[:isotropic]

# PSNR for peakvalue of 1
PSNR(x) = -10log10(sum(abs2.(x-I))/length(I))

# sparse-array TVD
@time x1, hist1 = tvd(y, λ, ρ; kw...)
psnr1 = PSNR(x1)
@printf "k=%d, PSNR1 = %.2f\n" hist1.k psnr1

# fft TVD
@time x2, hist2 = tvd_fft(y, λ, ρ; kw...)
psnr2 = PSNR(x2)
@printf "k=%d, PSNR2 = %.2f\n" hist2.k psnr2

# Primal-Dual Splitting TVD
@time x3, hist3 = tvd_pds(y, λ; θ=1, kw...)
psnr3 = PSNR(x3)
@printf "k=%d, PSNR3 = %.2f\n" hist3.k psnr3

# VAMP TVD
@time x4, hist4 = tvd_vamp(y, λ; kw...)
psnr4 = PSNR(x4)
@printf "k=%d, PSNR4 = %.2f\n" hist4.k psnr4

# showing images side-by-side
P = plot(axis=nothing, layout=(1,5), size=(2000,400))
imgv = tensor2img.([y, x1, x2, x3, x4])
psnrv  = [PSNR(y), psnr1, psnr2, psnr3, psnr4]
titlev = ["Noisy", "Sparse", "FFT", "PDS", "VAMP"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

iso = kw[:isotropic] ? "_iso" : "" # Julia has the ternary operator
#savefig("exfabio$iso.png")
gui() # show plot

