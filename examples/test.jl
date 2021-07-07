using TVDenoise
using Plots, Printf, TestImages, FileIO
using Flux, NNlib

outdir = "examples/out"
infn = readdir("/home/nikopj/doc/datasets/DAVIS/JPEGImages/480p/butterfly", join=true)
mkpath(outdir)

frame_load(path) = img2tensor(load(path)) |> x->Flux.unsqueeze(x,4)

video = Flux.stack([frame_load(infn[i]) for i ∈ 1:length(infn)], 1)
@info size(video)
video_n = video + 0.1*randn(size(video))

video_dn = zeros(size(video))
for i ∈ 1:length(infn)
	print("Processing frame "*@sprintf("%02d",i)*"...\r")
	x, hist = tvd_pds(video_n[i,:,:,:,1], 0.2, 0.5,0.5; isotropic=true, maxit=100, tol=1e-3, verbose=false)
	video_dn[i,:,:,:,1] = x
end
println("\nDone.")

for i ∈ 1:length(infn)
	outfn = @sprintf("%02d",i)*".jpg" |> x->joinpath(outdir,x)
	println(outfn)
	save(outfn, tensor2img(video_dn[i,:,:,:,1]))
end
