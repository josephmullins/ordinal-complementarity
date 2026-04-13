# Mixture-of-Monotone-Functions Parametric Model

## Setup

Object of interest: the technology of skill formation P(θ' | θ, I), where θ ∈ {1,...,r_θ} and I ∈ {1,...,r_I}.

Survivor function: F̄(k|θ,I) = P(θ' > k | θ, I) for k = 1,...,r_θ-1.

Valid distribution requires: 1 ≥ F̄(1|θ,I) ≥ F̄(2|θ,I) ≥ ... ≥ F̄(r_θ-1|θ,I) ≥ 0.

Complementarity (supermodularity): for all θ' > θ, I' > I, and all k:
F̄(k|θ',I') + F̄(k|θ,I) ≥ F̄(k|θ',I) + F̄(k|θ,I')

## Current parametric models

All current models use an ordered cutpoint + link function structure:

F̄(k|θ,I) = 1 - Φ(c_k - η(θ,I))

with c_1 < c_2 < ... < c_{r_θ-1} and Φ the normal CDF.

- **Parsimonious LPM**: η(θ,I) = γ₁I + γ₂θ + γ₃θI. Complementarity iff γ₃ ≥ 0.
- **Flexible LPM**: η(θ,I) = γ₀(θ) + γ₁(θ)·I. Complementarity iff γ₁(θ) non-decreasing.
- **Ordered probit**: η(θ,I) = β₁θ + β₂I + β₃θI. β₃ ≥ 0 necessary but not sufficient.

All linear in I for each θ. Only nonlinearity is the link function Φ.

## Three model classes

We consider three classes of models that generalize the ordered probit, ordered by increasing flexibility and decreasing interpretability.

### Model 1: Mixture of ordered probits with shared cutpoints (recommended)

```
F̄(k|θ,I) = Σ_h w_h(θ) · [1 - Φ(c_k - a_h · I)]
```

Each component h is a valid ordered probit survivor function with:
- Shared cutpoints c_1 < c_2 < ... < c_{r_θ-1} (increasing, CDF convention)
- Component-specific slope a_h > 0 (enforce via a_h = exp(ã_h))
- θ-dependent mixing weights w_h(θ) ≥ 0 with Σ_h w_h(θ) = 1 (softmax)

**Why this works**: Each component [1 - Φ(c_k - a_h·I)] is a valid survivor function (decreasing in k, in (0,1)) because the ordered cutpoints and Φ guarantee it. A convex combination of valid survivor functions is a valid survivor function. No clamping needed.

**Why softmax is required here**: Unlike Models 2–3, the weights here must sum to 1 because the output is a *direct mixture of distributions*, not an index passed through a link function. If the weights don't sum to 1, the mixture is not a valid probability distribution.

**Why shared cutpoints**: The cutpoints c_k determine the baseline distribution shape (where probability mass sits when I has no effect). Sharing them across components means the components differ only in their I-sensitivity a_h, not in their baseline shape. This is parsimonious and interpretable: the components represent different "channels" of investment response, all operating on the same ordinal scale. Component-specific cutpoints c_{h,k} could be added later if needed (adds (K-1)(r_θ-1) parameters).

**Parameter count**:

| Component | Count |
|-----------|-------|
| Cutpoints | r_θ - 1 (shared, increasing, reparameterized via log-diffs) |
| Slopes | K (one ã_h per component) |
| Mixing weights | (K-1) × r_θ (softmax, λ_{1,θ} = 0 for identification) |
| **Total** | **(r_θ - 1) + K + (K-1)·r_θ** |

Examples with r_θ = 3:

| K | Params | Comparison |
|---|--------|------------|
| 1 | 2 + 1 + 0 = 3 | Ordered probit: 5 |
| 2 | 2 + 2 + 3 = 7 | Flexible LPM: 8 |
| 3 | 2 + 3 + 6 = 11 | — |

Note: K = 1 nests the ordered probit (with only β₂, no β₁ or β₃). The full ordered probit adds a θ-dependent shift β₁·θ + β₃·θ·I to the index; with K ≥ 2, the mixture weights w_h(θ) absorb this role.

**Complementarity**: Since Φ is concave on (0,∞) and convex on (-∞,0), supermodularity of the index does not directly imply supermodularity of the survivor function. Must test on the resulting Π matrix using the existing nonparametric S-matrix test.

**Intuition**: Order components by slope (a_1 < a_2 < ... < a_K). Complementarity is favored when higher θ puts more weight on steeper components — i.e., when the mixing weights exhibit a FOSD shift toward higher-slope components as θ increases.

**Nesting**: With K = 1, the model reduces to P(θ' ≤ k | I) = Φ(c_k - a·I), which is the ordered probit without θ-dependence. K = 1 cannot produce different distributions at different θ since there's only one component. With K ≥ 2, θ-dependence enters through the mixing weights.


### Model 2: Sigmoid index with separable basis functions

```
F̄(k|θ,I) = σ(c_k + Σ_h w_h(θ) · g_h(I))
```

where:
- σ: ℝ → (0,1) is a sigmoid link (logistic or Φ)
- c_1 > c_2 > ... > c_{r_θ-1} are ordered cutpoints (decreasing for decreasing F̄)
- g_h(I) are basis functions, each non-decreasing in I
- w_h(θ) ≥ 0 are non-negative weights depending on θ

**Valid distributions**: σ strictly increasing + decreasing cutpoints → F̄ strictly decreasing in k → positive PMF. No clamping needed.

**Basis function choices**:
- **(a) Sigmoid**: g_h(I) = σ(a_h·I + b_h), a_h > 0. 2 params per basis, automatically monotone.
- **(b) Scaled linear**: g_h(I) = α_h·I, α_h > 0.
- **(c) Free monotone**: g_h(i) = g_h(1) + Σ_{j=2}^{i} exp(d̃_{h,j}). r_I - 1 params per basis.

**Weight options**:
- **Softmax** (Σ w_h = 1): interpretable as mixing proportions, but bounds the index in (0,1) with sigmoid basis, limiting the maximum investment effect to σ(c_k+1) - σ(c_k) ≈ 0.23.
- **Exp** (w_h = exp(λ_h), unnormalized): unbounded index, full approximation power for monotone functions. Recommended for this model.

**Parameter count** (sigmoid basis, exp weights, r_θ = 3):

| K | Params | Comparison |
|---|--------|------------|
| 1 | 2 + 2 + 3 = 7 | Ordered probit: 5 |
| 2 | 2 + 4 + 6 = 12 | Flexible LPM: 8 |
| 3 | 2 + 6 + 9 = 17 | — |

Formula: (r_θ - 1) + 2K + K·r_θ.


### Model 3: Non-separable monotone neural network

```
F̄(k|θ,I) = σ(c_k + Σ_h w_h · σ(α_{h,1}·θ + α_{h,2}·I + b_h))
```

with w_h > 0 (via exp), α_{h,2} > 0 (via exp). Each hidden unit takes both θ and I as inputs.

**Valid distributions**: same mechanism as Model 2 (σ + decreasing cutpoints).

**Parameter count** (r_θ = 3):

| K | Params |
|---|--------|
| 1 | 2 + 4 = 6 |
| 2 | 2 + 8 = 10 |
| 3 | 2 + 12 = 14 |

Formula: (r_θ - 1) + 4K (each unit has ã_{h,1}, ã_{h,2}, b_h, log w_h).


### Comparison of the three models

| | Model 1: Mixture of probits | Model 2: Sigmoid index | Model 3: Non-separable NN |
|-|---------------------------|----------------------|--------------------------|
| Specification | Σ w_h(θ)·[1-Φ(c_k - a_h·I)] | σ(c_k + Σ w_h(θ)·g_h(I)) | σ(c_k + Σ w_h·σ(α_1θ + α_2I + b)) |
| Valid distribution via | convex combo of valid dists | σ + cutpoints | σ + cutpoints |
| Weights must | sum to 1 (softmax) | be non-negative (exp ok) | be non-negative (exp ok) |
| θ enters through | mixing weights only | mixing weights only | hidden units + weights |
| Monotonicity in I | automatic (each Φ monotone, w ≥ 0) | automatic (g_h monotone, w ≥ 0) | automatic (α₂ > 0, w ≥ 0) |
| Bounded effect concern | no (components are full distributions) | yes with softmax, no with exp | no |
| Params (r_θ=3, K=2) | **7** | 10-12 | 10 |
| Interpretability | high | medium | low |
| θ-I interaction | transparent (inspect weights) | transparent (inspect weights) | tangled inside hidden units |

**Recommendation**: Start with Model 1 (mixture of ordered probits with shared cutpoints). It is the most parsimonious, the most interpretable, and valid by construction without any approximation-theoretic arguments. If it proves too restrictive (e.g., shared cutpoints limit fit), move to component-specific cutpoints before switching to Model 2 or 3.


## Why separate θ from the basis functions?

All three models enforce monotonicity in I. Models 1 and 2 additionally enforce **separability**: the basis functions / components depend only on I, and θ enters only through the mixing weights. Model 3 allows θ inside the hidden units.

Separability is not needed for monotonicity. If θ enters the hidden units as σ(α₁·θ + α₂·I + b), the derivative w.r.t. I is Σ_h w_h·α_{h,2}·σ'(α_{h,1}·θ + α_{h,2}·I + b_h), which is positive when α_{h,2} > 0 and w_h > 0.

The reason for separability is **interpretability of the θ-I interaction** (complementarity). With separable models, the basis functions are fixed curves; θ only determines how much of each curve to use. You can inspect the weights to see how the I-response changes with θ. With non-separable models, θ deforms every hidden unit's response to I, making the interaction hard to decompose.


## Connection to neural networks

### How a generic neural network approximates the distribution

A single-hidden-layer network with K units computes:
```
f(k, θ, I) = β₀(k) + Σ_h v_h(k) · σ(α_{h,1}·θ + α_{h,2}·I + α_{h,3}·k + b_h)
```
This can approximate any continuous function of (θ,I,k) as K → ∞ (universal approximation).

But a generic network doesn't produce valid distributions. To fix this, use the **ordered output approach**:
```
F̄(k|θ,I) = σ(c_k + μ(θ,I))
```
where c_1 > c_2 > ... and μ(θ,I) is a standard neural network:
```
μ(θ,I) = β₀ + Σ_h v_h · σ(α_{h,1}·θ + α_{h,2}·I + b_h)
```
This is an ordered logit where the linear index is replaced by a neural network. Valid distributions come from σ + ordered cutpoints.

### Our models as constrained versions

All three models are constrained versions of this general architecture:

**1. Monotonicity in I**: In the general network, the output weights v_h are unconstrained real numbers — positive, negative, or zero. Negative weights allow hidden units to cancel each other, producing non-monotone functions. Our models constrain w_h ≥ 0 and (in Models 2–3) a_h or α_{h,2} > 0, so the index is non-decreasing in I.

**2. Separability** (Models 1–2): Hidden units depend only on I, not θ. θ enters only through the output-layer weights. This is a **generalized additive network**: θ and I affect the output through separate channels that interact only at the output layer.

**3. Distributional structure** (Model 1): The output is a direct mixture of distributions rather than a sigmoid of an index. This is more constrained but automatically valid and avoids the bounded-effect concern.

| | General NN | Model 3 | Model 2 (exp) | Model 1 |
|-|-----------|---------|---------------|---------|
| Hidden units | f(θ, I) | f(θ, I), α₂>0 | f(I) only | Φ(c_k - a·I) |
| Output weights | v_h ∈ ℝ | w_h > 0 | w_h(θ) > 0 | w_h(θ), Σ=1 |
| Monotone in I | no | yes | yes | yes |
| Separable | no | no | yes | yes |
| Valid dist via | extra structure | σ + cutpoints | σ + cutpoints | mixture |

### Interpretation of hidden units

In a standard NN, hidden units are uninterpretable. In Models 1–2, each component/basis function represents a **developmental channel** — a monotone pathway through which investment affects skills. In Model 1, each channel is a full ordered probit with its own I-sensitivity a_h. In Model 2, each channel is a sigmoid that "turns on" at a different level of I.

The mixing weights w_h(θ) determine how much each channel matters for a child at skill level θ. Complementarity = higher-skilled children benefit more from the responsive channels.

### Extension to additional covariates

The mixing weights can depend on additional state variables:
```
w_h(θ, θ^P, ω) = exp(λ_h(θ, θ^P, ω)) / Σ_{h'} exp(λ_{h'}(θ, θ^P, ω))
```
while basis functions / components stay functions of I alone. This keeps the investment channel interpretable and monotone while allowing "routing" to depend on the full state. In NN terms, this is a **hypernetwork** or **conditional computation** setup.


## Implementation plan (Model 1)

### New struct: `MixtureProbitTransition`

Add to `src/parametric_models.jl`:

```julia
struct MixtureProbitTransition
    c_raw::Vector{Float64}      # r_θ-1 cutpoint log-diffs (increasing, CDF convention)
    a_raw::Vector{Float64}      # K log-slopes (enforce a_h > 0 via exp)
    lambda::Matrix{Float64}     # (K-1) × r_θ softmax logits (row 1 normalized to 0)
    K::Int                      # number of mixture components
end
```

Note: uses existing `_pack_cutpoints`/`_unpack_cutpoints` (increasing order, CDF convention) — same as `OrderedProbitTransition`. No need for `_pack_survivor_cutoffs`.

### `pack` / `unpack`

```julia
function pack_mixture_probit(spec::MixtureProbitTransition)
    vcat(_pack_cutpoints(spec.c_raw), spec.a_raw, vec(spec.lambda))
end

function unpack_mixture_probit(vec::AbstractVector, r_θ::Int, K::Int)
    n_cut = r_θ - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    a_raw = vec[n_cut+1:n_cut+K]
    n_lambda = (K - 1) * r_θ
    lambda = reshape(vec[n_cut+K+1:n_cut+K+n_lambda], K - 1, r_θ)
    MixtureProbitTransition(c, a_raw, lambda, K)
end
```

### `build` → Π matrix

```julia
function _build_mixture_probit_pi(vec::AbstractVector{T}, r_θ::Int, r_I::Int, K::Int;
                                   I_grid::AbstractVector=1.0:r_I) where T
    n_cut = r_θ - 1
    c = _unpack_cutpoints(vec[1:n_cut])
    a = exp.(vec[n_cut+1:n_cut+K])

    # Softmax mixing weights: w[h, θ]
    n_lambda = (K - 1) * r_θ
    lambda_free = reshape(vec[n_cut+K+1:n_cut+K+n_lambda], K - 1, r_θ)

    Π = zeros(T, r_θ, r_θ, r_I)
    for kI in 1:r_I, kθ_idx in 1:r_θ
        # Compute softmax weights for this θ
        logits = vcat(zero(T), lambda_free[:, kθ_idx])
        max_l = maximum(logits)
        exp_l = exp.(logits .- max_l)
        w = exp_l ./ sum(exp_l)

        # Mixture survivor: F̄(k|θ,I) = Σ_h w_h · [1 - Φ(c_k - a_h·I)]
        for k in 1:r_θ-1
            S_bar_k = zero(T)
            for h in 1:K
                S_bar_k += w[h] * (one(T) - Φ(c[k] - a[h] * I_grid[kI]))
            end
            # Will convert to PMF below
        end

        # Build full survivor then PMF
        S_bar = zeros(T, r_θ - 1)
        for k in 1:r_θ-1
            for h in 1:K
                S_bar[k] += w[h] * (one(T) - Φ(c[k] - a[h] * I_grid[kI]))
            end
        end

        # PMF from survivor
        pmf = Vector{T}(undef, r_θ)
        pmf[1] = one(T) - S_bar[1]
        for k in 2:r_θ-1
            pmf[k] = S_bar[k-1] - S_bar[k]
        end
        pmf[r_θ] = S_bar[r_θ-1]

        # Floor and renormalize for numerical safety
        pmf .= max.(pmf, T(1e-12))
        pmf ./= sum(pmf)
        Π[:, kθ_idx, kI] = pmf
    end
    return Π
end
```

### M-step

Uses `_transition_negll_precomputed` (already exists) with the mixture probit build function:

```julia
function transition_negll_mixture_probit_fast(vec::AbstractVector{T},
                                               Q_sum::Array{Float64,3},
                                               r_θ::Int, r_I::Int, K::Int;
                                               I_grid::AbstractVector=1.0:r_I) where T
    Π = _build_mixture_probit_pi(vec, r_θ, r_I, K; I_grid)
    return _transition_negll_precomputed(Π, Q_sum)
end

function parametric_mstep_transition(spec::MixtureProbitTransition,
                                      Q::Array{Float64,4},
                                      r_θ::Int, r_I::Int;
                                      I_grid::AbstractVector=collect(1.0:r_I),
                                      options=DEFAULT_OPTIM_OPTIONS)
    Q_sum = _precompute_Q_sum(Q)
    x0 = pack_mixture_probit(spec)
    obj = x -> transition_negll_mixture_probit_fast(x, Q_sum, r_θ, r_I, spec.K; I_grid)
    result = optimize(obj, x0, LBFGS(), options; autodiff=:forward)
    new_spec = unpack_mixture_probit(Optim.minimizer(result), r_θ, spec.K)
    Π = _build_mixture_probit_pi(Optim.minimizer(result), r_θ, r_I, spec.K; I_grid)
    return new_spec, Π
end
```

### Initialization

```julia
function initialize_mixture_probit(r_θ::Int, K::Int;
                                    rng::AbstractRNG=Random.default_rng())
    n_cut = r_θ - 1
    # Increasing cutpoints (CDF convention)
    c = collect(range(-1.0, 1.0; length=n_cut)) .+ 0.02 .* randn(rng, n_cut)
    sort!(c)
    # Log-slopes: spread them out so components have different sensitivities
    a_raw = collect(range(-0.5, 0.5; length=K)) .+ 0.1 .* randn(rng, K)
    # Softmax logits: near-uniform initial mixing
    lambda = 0.05 .* randn(rng, K - 1, r_θ)
    MixtureProbitTransition(c, a_raw, lambda, K)
end
```

### Integration with existing code

1. Add `MixtureProbitTransition` dispatch to `parametric_mstep_transition`
2. The E-step is unchanged — it only sees the Π matrix
3. Inference: use `get_score_parametric` with the Jacobian of pack → Π (ForwardDiff handles this)
4. Complementarity test: apply existing `get_S` and `get_test_stat` to the resulting Π

### New estimation script

Create `scripts/run_estimation_mixture.jl` following the pattern of `run_estimation_parametric.jl`:
- Same data preparation (focused sources, HOME-SF binary)
- Replace `OrderedProbitTransition` with `MixtureProbitTransition`
- Keep `OrderedProbitEmission` for emissions (or use nonparametric)
- Start with K=2, compare with K=1 (which should roughly match ordered probit results)

### Testing

1. **K=1 nesting**: verify that K=1 results are close to ordered probit (without θ-dependence in index)
2. **Simulated data**: generate from a known mixture, check recovery
3. **Monotonicity**: verify Π is monotone in I (FOSD) for all estimated models
4. **Real data**: run on all 6 age groups, compare log-likelihoods and complementarity test results

### Things to watch out for

- **Identification with small r_I**: with r_I = 3, each component's Φ(c_k - a_h·I) is evaluated at only 3 I-values. K = 2 or 3 is fine; K > r_I may be overparameterized.
- **Local optima**: mixture models have more local optima than single-index models. Use multiple random restarts (already in the EM framework).
- **Label switching**: components are exchangeable (permuting h with corresponding w_h gives the same model). Doesn't affect estimation but matters for interpretation. Post-hoc order by slope a_h.
- **Cutpoint convention**: Model 1 uses CDF convention (c_1 < c_2, increasing), same as `OrderedProbitTransition`. Use `_pack_cutpoints`/`_unpack_cutpoints`, NOT `_pack_survivor_cutoffs`.
