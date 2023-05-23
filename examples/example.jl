using TVDenoise
using Plots, Printf, TestImages, FileIO
#=
Example TV denoising!
=#

# generate noisy image
img = testimage("fabio_color_512")
I = img2tensor(img)
y = I + 0.1*randn(size(I))
@info size(y)

# TVD parameters
λ = 0.12; ρ = 2
kw = Dict(:isotropic=>true, :maxit=>15, :tol=>1e-3, :verbose=>true)
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
@time x3, hist3 = tvd_pds(y, λ; θ=0, kw...)
psnr3 = PSNR(x3)
@printf "k=%d, PSNR3 = %.2f\n" hist3.k psnr3

# VAMP TVD
@time x4, hist4 = tvd_vamp(y, λ; kw...)
psnr4 = PSNR(x4)
@printf "k=%d, PSNR4 = %.2f\n" hist4.k psnr4

# TGV-2 TVD
@time x5, hist5 = tgv_pds(y, λ, 0.25; θ=0, kw...)
psnr5 = PSNR(x5)
@printf "k=%d, PSNR5 = %.2f\n" hist5.k psnr5

# MGProx-PDS TVD
L = log2(minimum(size(y)[1:2])) - 4 |> floor |> Int
@show L
@time x6, hist6 = mg_tvd_pds(y, λ, L; θ=0, α=0.2, n_inner=10, n_coarse=50, linesearch=false, kw...)
psnr6 = PSNR(x6)
@printf "k=%d, PSNR6 = %.2f\n" hist6.k psnr6

# showing images side-by-side
P = plot(axis=nothing, layout=(2,3), size=(1200,800))
imgv = tensor2img.([x1, x2, x3, x4, x5, x6])
psnrv  = [psnr1, psnr2, psnr3, psnr4, psnr5, psnr6]
titlev = ["Sparse", "FFT", "PDS", "VAMP", "TGV2-PDS", "MGProx-PDS"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

iso = kw[:isotropic] ? "_iso" : "" # Julia has the ternary operator
#savefig("exfabio$iso.png")
gui() # show plot

