using Distributions, Plots
function myqqplot(obs,F⁰,title)
    nobs=length(obs)
    sort!(obs)
    quantiles⁰ = [quantile(F⁰,i/nobs) for i in 1:nobs]
    # Note that only n-1 points may be plotted, as quantile(F⁰,1) may be inf
    plot(quantiles⁰, obs, seriestype=:line, xlabel="Theoretical Quantiles", ylabel = "Sample Quantiles", title=title, label="" )
    plot!(obs,obs,label="")
end
obs = rand(Normal(0,1),1000)
F⁰=Normal(0,1)
myqqplot(obs,F⁰,"Normal QQ-plot ")
