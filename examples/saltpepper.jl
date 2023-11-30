using TVDenoise
using Plots, Printf, TestImages, Images, FileIO
#=
Example TV denoising (with Salt and Pepper noise)!
=#

# generate noisy image
img = testimage("fabio_color_512")
#img = load("examples/cameraman.jpg")
x = img2tensor(img)
y = saltpepper(x, 0.5)
@info size(y)

# TVD parameters
λ = 1.5
kw = Dict(:ℓ1=>true, :θ=>0.0, :isotropic=>true, :tol=>5e-3, :verbose=>false)
@info kw[:isotropic]

# PSNR for peakvalue of 1
PSNR(x̂) = -10log10(sum(abs2, x-x̂)/length(x))

# Primal-Dual Splitting TVD
@time x̂1, hist1 = tvd_pds(y, λ; maxit=500, kw...)
psnr1 = PSNR(x̂1)
@printf "k=%d, PSNR = %.2f\n" hist1.k psnr1

# Primal-Dual Splitting TGV-2
@time x̂2, hist2 = tgv_pds(y, λ, 0.2; maxit=500, kw...)
psnr2 = PSNR(x̂2)
@printf "k=%d, PSNR = %.2f\n" hist2.k psnr2

# MGProx-PDS TVD
L = log2(minimum(size(y)[1:2])) - 2 |> Int
@show L
@time x̂3, hist3 = mg_tvd_pds(y, λ, L; α=0.1, n_inner=1, n_coarse=200, linesearch=true, maxit=50, kw...)
psnr3 = PSNR(x̂3)
@printf "k=%d, PSNR3 = %.2f\n" hist3.k psnr3

# showing images side-by-side
P = plot(axis=nothing, layout=(1,4), size=(1600,400))
imgv = tensor2img.([y, x̂1, x̂2, x̂3])
psnrv  = [PSNR(y), psnr1, psnr2, psnr3]
titlev = ["Noisy", "PDS", "TGV-PDS", "MGProx-PDS"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

#iso = kw[:isotropic] ? "_iso" : ""
#savefig("saltpepper_fabio$iso.png")
gui() # show plot

