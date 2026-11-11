# Toy Monte Carlo: 95% CI coverage for t = max(μ1, μ2) estimated by the max of
# two sample means — percentile vs basic bootstrap vs numerical delta method.
using Random, Statistics, Printf
Random.seed!(1)
const n, B, R = 400, 999, 2000

function sim(μ)
    cov = zeros(3); t = maximum(μ); ε = n^(-1/3)
    for r in 1:R
        X = randn(n, 2) .+ permutedims(μ)
        m̂ = vec(mean(X, dims = 1)); t̂ = maximum(m̂)
        tb = zeros(B); Db = zeros(B)
        for b in 1:B
            mb = vec(mean(X[rand(1:n, n), :], dims = 1))
            tb[b] = maximum(mb)
            Db[b] = (maximum(m̂ .+ ε * sqrt(n) .* (mb .- m̂)) - t̂) / ε
        end
        S = sqrt(n) .* (tb .- t̂)
        cis = ((quantile(tb, .025), quantile(tb, .975)),
               (t̂ - quantile(S, .975) / sqrt(n), t̂ - quantile(S, .025) / sqrt(n)),
               (t̂ - quantile(Db, .975) / sqrt(n), t̂ - quantile(Db, .025) / sqrt(n)))
        for k in 1:3
            cov[k] += cis[k][1] <= t <= cis[k][2]
        end
    end
    cov ./ R
end

for (lab, μ) in (("well separated μ=(0,-0.5) ", [0.0, -0.5]),
                 ("near tie       μ=(0,-0.05)", [0.0, -0.05]),
                 ("exact tie      μ=(0,0)    ", [0.0, 0.0]))
    c = sim(μ)
    @printf("%s  coverage of 95%% CI:  percentile %.3f   basic %.3f   NDM %.3f\n", lab, c...)
end
