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
λ = 0.1
ρ = 2
kw = Dict(:isotropic=>true, :maxit=>200, :tol=>1e-2, :verbose=>true)
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

# showing images side-by-side
P = plot(axis=nothing, layout=(1,3), size=(1200,400))
imgv = tensor2img.([y, x1, x2])
psnrv  = [PSNR(y), psnr1, psnr2]
titlev = ["Noisy", "Sparse", "FFT"]

for i=1:length(P)
	plot!(P[i], imgv[i])
	title!(P[i], @sprintf "%s: %.2f dB" titlev[i] psnrv[i])
end

iso = kw[:isotropic] ? "_iso" : "" # Julia has the ternary operator
#savefig("exfabio$iso.png")
gui() # show plot

