# Fixed, content-based groupings of HOME investment items (2026-11-17).
# Each group present in a period is replaced by ONE composite measure: the mean of the
# child's observed items in the group (each scaled to [0,1]), cut into ≤ NCAT ordered
# categories at quantiles. Groups with a single item present keep that item unchanged.
# Motivation: items within a HOME sub-scale are correlated beyond the latent investment
# level (local dependence), which made latent investment collapse onto single items.
const INVESTMENT_GROUPS = [
    ("Reading & materials", [:HowOftenRead, :HowManyBooks, :HowManyToysA, :HowManyToysB, :HowManyZines,
                             :RecordTapeCD, :EnjoyRead, :DailyNewspaper]),
    ("Teaching",            [:LearnNumbers, :LearnAlphabet, :LearnColors, :LearnShapes]),
    ("Outings",             [:OutOfHouse, :HowOftenGrocery, :HowOftenOuting, :HowOftenMuseum, :HowOftenTheater]),
    ("Family & father",     [:HowOftenFamily, :HowOftenOutdoors, :HowOftenFather, :DiscussTV]),
    ("School involvement",  [:GradesTeacher, :GradesTalk, :GradesHelp, :GradesStudy]),
    ("Enrichment",          [:ChildMusic, :SpecialActivities, :Hobbies]),
    ("Chores & routines",   [:HowOftenClean, :HowOftenChores, :HowOftenTime]),
]

# item names in the panel may carry suffixes (e.g. "HowManyToysA_q4")
const COMPOSITE_COMPRESS = Ref(false)   # true: drop empty composite categories (2026-11-18)
_item_base(name) = Symbol(replace(String(name), r"_q\d+$" => ""))

function _group_composite(X, κ, idx; ncat = 5)
    N = length(X[1]); sc = fill(NaN, N)
    for n in 1:N
        v = [(X[m][n] - 1) / (κ[m] - 1) for m in idx if X[m][n] > 0]
        isempty(v) || (sc[n] = mean(v))
    end
    obs = findall(!isnan, sc); u = sort(unique(sc[obs]))
    cuts = length(u) <= ncat ? u[2:end] : unique(quantile(sc[obs], (1:ncat-1) ./ ncat))
    x = zeros(Int, N)
    for n in obs; x[n] = 1 + count(c -> sc[n] >= c, cuts); end
    COMPOSITE_COMPRESS[] ? compress_categories(x) : (x, maximum(x))   # a cut at the minimum left category 1 empty
end

"""Return (X_I, κ_I, measure_names_I) with each content group replaced by one composite."""
function group_investment_measures(X_I, κ_I, names_I; ncat = 5)
    T = length(X_I); Xo = similar(X_I); κo = similar(κ_I); no = similar(names_I)
    for t in 1:T
        base = _item_base.(names_I[t]); used = falses(length(base))
        Xt = Vector{Int}[]; κt = Int[]; nt = String[]
        for (gname, items) in INVESTMENT_GROUPS
            idx = findall(b -> b in items, base)
            isempty(idx) && continue
            used[idx] .= true
            if length(idx) == 1
                push!(Xt, X_I[t][idx[1]]); push!(κt, κ_I[t][idx[1]]); push!(nt, String(names_I[t][idx[1]]))
            else
                x, k = _group_composite(X_I[t], κ_I[t], idx; ncat)
                push!(Xt, x); push!(κt, k); push!(nt, "[" * gname * "]")
            end
        end
        for m in findall(!, used)
            push!(Xt, X_I[t][m]); push!(κt, κ_I[t][m]); push!(nt, String(names_I[t][m]))
        end
        Xo[t], κo[t], no[t] = Xt, κt, nt
    end
    Xo, κo, no
end
