using TVDenoise
using Plots, Printf, TestImages, Images, FileIO
#=
Example TV denoising (with Salt and Pepper noise)!
=#

# generate noisy image
#img = testimage("fabio_color_256")
img = load("cameraman.jpg")
x = img2tensor(img)
y = saltpepper(x, 0.2)
@info size(y)

# TVD parameters
λ = 1
kw = Dict(:ℓ1=>true, :θ=>0, :isotropic=>true, :maxit=>2000, :tol=>1e-3, :verbose=>true)
@info kw[:isotropic]

# PSNR for peakvalue of 1
PSNR(x̂) = -10log10(sum(abs2, x-x̂)/length(x))

# Primal-Dual Splitting TVD
@time x̂, hist = tvd_pds(y, λ; kw...)
psnr = PSNR(x̂)
@printf "k=%d, PSNR = %.2f\n" hist.k psnr

# showing images side-by-side
P = plot(axis=nothing, layout=(1,2), size=(800,400))
imgv = tensor2img.([y, x̂])
psnrv  = [PSNR(y), psnr]
titlev = ["Noisy", "PDS"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

#iso = kw[:isotropic] ? "_iso" : ""
#savefig("saltpepper_fabio$iso.png")
gui() # show plot

