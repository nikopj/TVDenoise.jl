using TVDenoise
using Plots, Printf, TestImages, FileIO
#=
Example TV denoising!
=#

# generate noisy image
img = testimage("fabio_color_256")
I = img2tensor(img)
y = I + 0.1*randn(size(I))
@info size(y)

# TVD parameters
λ = 0.12; ρ = 2; θ = 0.8
kw = Dict(:isotropic=>true, :maxit=>20, :tol=>5e-3, :verbose=>false)
@info kw[:isotropic]

# PSNR for peakvalue of 1
PSNR(x) = -10log10(sum(abs2.(x-I))/length(I))

# sparse-array TVD
@time x1, hist1 = tvd(y, λ, ρ; kw...)
psnr1 = PSNR(x1)
@printf "k=%d, ADMM-SPARSE PSNR1 = %.2f\n" hist1.k psnr1

# fft TVD
@time x2, hist2 = tvd_fft(y, λ, ρ; kw...)
psnr2 = PSNR(x2)
@printf "k=%d, ADMM-FFT PSNR2 = %.2f\n" hist2.k psnr2

# Primal-Dual Splitting TVD
@time x3, hist3 = tvd_pds(y, λ; θ=θ, kw...)
psnr3 = PSNR(x3)
@printf "k=%d, PDS PSNR3 = %.2f\n" hist3.k psnr3

# VAMP TVD
@time x4, hist4 = tvd_vamp(y, λ; kw...)
psnr4 = PSNR(x4)
@printf "k=%d, VAMP PSNR4 = %.2f\n" hist4.k psnr4

# TGV-2 TVD
@time x5, hist5 = tgv_pds(y, λ, 0.25; θ=θ, kw...)
psnr5 = PSNR(x5)
@printf "k=%d, TGV2 PSNR5 = %.2f\n" hist5.k psnr5

# MGProx-PDS TVD
L = log2(minimum(size(y)[1:2])) - 2 |> floor |> Int
@show L
@time x6, hist6 = mg_tvd_pds(y, λ, L; θ=θ, α=0.1, t=0, n_inner=2, n_coarse=20, linesearch=false, kw...)
psnr6 = PSNR(x6)
@printf "k=%d, MGProx PDS PSNR6 = %.2f\n" hist6.k psnr6

# showing images side-by-side
P = plot(axis=nothing, layout=(2,3), size=(1200,800))
imgv = tensor2img.([x1, x2, x3, x4, x5, x6])
psnrv  = [psnr1, psnr2, psnr3, psnr4, psnr5, psnr6]
titlev = ["Sparse", "FFT", "PDS", "VAMP", "TGV2-PDS", "MGProx-PDS"]

# imgv = tensor2img.([y, x3, x6])
# psnrv = [PSNR(y), psnr3, psnr6]
# titlev = ["Noisy", "PDS", "MGProx-PDS"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

iso = kw[:isotropic] ? "_iso" : "" # Julia has the ternary operator
#savefig("exfabio$iso.png")
gui() # show plot

