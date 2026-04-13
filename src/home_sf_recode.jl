# ─────────────────────────────────────────────────────────────────
# HOME-SF (Short Form) Binary Recoding Schema
#
# Based on NLS NLSY79 Children Codebook Supplement:
#   Appendix A-2a (Under Age 3), A-2b (Ages 3-5),
#   A-2c (Ages 6-9), A-2d (Ages 10-14)
#
# Each item is recoded to binary (0/1) following official rules.
# Scale: :C = Cognitive Stimulation, :E = Emotional Support
# Only self-report items are included (interviewer observations
# CSOB-* are not available in HOME_panel.csv).
# ─────────────────────────────────────────────────────────────────

"""
    HOMESF_SCHEMA

Dictionary mapping HOME Part symbols (`:PartA`, `:PartB`, `:PartC`, `:PartD`)
to vectors of recoding rules. Each rule is a named tuple:
  - `column::Symbol` — column name in HOME_panel.csv
  - `scale::Symbol` — `:C` (Cognitive Stimulation) or `:E` (Emotional Support)
  - `one_if::Function` — `f(v::Int) -> Bool`; returns `true` when raw value maps to 1
"""
const HOMESF_SCHEMA = Dict{Symbol, Vector{@NamedTuple{
    column::Symbol, scale::Symbol, one_if::Function}}}(

    # ─────────────────────────────────────────────────────────────
    # Part A: Children Under Age 3
    # ─────────────────────────────────────────────────────────────
    :PartA => [
        # ── Cognitive Stimulation (C) ── 7 items
        (; column=:OutOfHouse,       scale=:C, one_if = v -> v in (6, 7)),
        (; column=:HowManyBooks,     scale=:C, one_if = v -> v in (3, 4)),
        (; column=:HowOftenRead,     scale=:C, one_if = v -> v in (5, 6)),
        (; column=:HowOftenGrocery,  scale=:C, one_if = v -> v == 1),
        (; column=:HowManyToysA,     scale=:C, one_if = v -> v >= 1),
        (; column=:HowManyToysB,     scale=:C, one_if = v -> v >= 1),
        (; column=:MotherBelief,     scale=:C, one_if = v -> v in (1, 2)),

        # ── Emotional Support (E) ── 3 self-report items
        (; column=:HowOftenEats,     scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenTalks,    scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenSpank,    scale=:E, one_if = v -> v in (0, 1)),
    ],

    # ─────────────────────────────────────────────────────────────
    # Part B: Children Ages 3-5
    # ─────────────────────────────────────────────────────────────
    :PartB => [
        # ── Cognitive Stimulation (C) ── 10 items
        (; column=:HowOftenRead,     scale=:C, one_if = v -> v in (5, 6)),
        (; column=:HowManyBooks,     scale=:C, one_if = v -> v == 4),
        (; column=:HowManyZines,     scale=:C, one_if = v -> v >= 2),
        (; column=:RecordTapeCD,     scale=:C, one_if = v -> v == 1),
        (; column=:LearnNumbers,     scale=:C, one_if = v -> v == 1),
        (; column=:LearnAlphabet,    scale=:C, one_if = v -> v == 1),
        (; column=:LearnColors,      scale=:C, one_if = v -> v == 1),
        (; column=:LearnShapes,      scale=:C, one_if = v -> v == 1),
        (; column=:HowOftenOuting,   scale=:C, one_if = v -> v in (3, 4, 5)),
        (; column=:HowOftenMuseum,   scale=:C, one_if = v -> v >= 2),

        # ── Emotional Support (E) ── 6 self-report items
        # Note: HitHitBack and HitSpanking are reverse-coded. If either is
        # checked (value==1), the entire B08B discipline series scores 0.
        (; column=:FoodChoice,       scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HoursTVOn,        scale=:E, one_if = v -> v <= 4),
        (; column=:HowOftenEats,     scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenSpank,    scale=:E, one_if = v -> v in (0, 1)),
        (; column=:HitHitBack,       scale=:E, one_if = v -> v == 0),     # reverse-coded
        (; column=:HitSpanking,      scale=:E, one_if = v -> v == 0),     # reverse-coded
    ],

    # ─────────────────────────────────────────────────────────────
    # Part C: Children Ages 6-9
    # ─────────────────────────────────────────────────────────────
    :PartC => [
        # ── Cognitive Stimulation (C) ── 10 items
        (; column=:HowManyBooks,     scale=:C, one_if = v -> v == 4),
        (; column=:HowOftenRead,     scale=:C, one_if = v -> v in (5, 6)),
        (; column=:ChildMusic,       scale=:C, one_if = v -> v == 1),
        (; column=:DailyNewspaper,   scale=:C, one_if = v -> v == 1),
        (; column=:EnjoyRead,        scale=:C, one_if = v -> v in (1, 2)),
        (; column=:Hobbies,          scale=:C, one_if = v -> v == 1),
        (; column=:SpecialActivities,scale=:C, one_if = v -> v == 1),
        (; column=:HowOftenMuseum,   scale=:C, one_if = v -> v >= 2),
        (; column=:HowOftenTheater,  scale=:C, one_if = v -> v >= 2),
        (; column=:DiscussTV,        scale=:C, one_if = v -> v == 1),

        # ── Emotional Support (E) ── 10 self-report items
        (; column=:HowOftenBed,      scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenRoom,     scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenSpills,   scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenBathe,    scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenPickUp,   scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenFamily,   scale=:E, one_if = v -> v in (4, 5)),
        (; column=:HowOftenFather,   scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenOutdoors, scale=:E, one_if = v -> v in (1, 2, 3)),
        (; column=:HowOftenEats,     scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenSpank,    scale=:E, one_if = v -> v in (0, 1)),
        # Tantrum discipline series: TantrumSpanking is reverse-coded
        (; column=:TantrumSpanking,  scale=:E, one_if = v -> v == 0),     # reverse-coded
    ],

    # ─────────────────────────────────────────────────────────────
    # Part D: Children Ages 10-14
    # ─────────────────────────────────────────────────────────────
    :PartD => [
        # ── Cognitive Stimulation (C) ── 9 items (HowOftenRead drops)
        (; column=:HowManyBooks,     scale=:C, one_if = v -> v == 4),
        (; column=:ChildMusic,       scale=:C, one_if = v -> v == 1),
        (; column=:DailyNewspaper,   scale=:C, one_if = v -> v == 1),
        (; column=:EnjoyRead,        scale=:C, one_if = v -> v in (1, 2)),
        (; column=:Hobbies,          scale=:C, one_if = v -> v == 1),
        (; column=:SpecialActivities,scale=:C, one_if = v -> v == 1),
        (; column=:HowOftenMuseum,   scale=:C, one_if = v -> v >= 2),
        (; column=:HowOftenTheater,  scale=:C, one_if = v -> v >= 2),
        (; column=:DiscussTV,        scale=:C, one_if = v -> v == 1),

        # ── Emotional Support (E) ── 12 self-report items
        (; column=:HowOftenBed,      scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenRoom,     scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenPickUp,   scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenClean,    scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenChores,   scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenTime,     scale=:E, one_if = v -> v >= 2),
        (; column=:HowOftenFamily,   scale=:E, one_if = v -> v in (4, 5)),
        (; column=:HowOftenFather,   scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenOutdoors, scale=:E, one_if = v -> v in (1, 2, 3)),
        (; column=:HowOftenEats,     scale=:E, one_if = v -> v in (1, 2)),
        (; column=:HowOftenSpank,    scale=:E, one_if = v -> v in (0, 1)),
        # Tantrum discipline series: TantrumSpanking is reverse-coded
        (; column=:TantrumSpanking,  scale=:E, one_if = v -> v == 0),     # reverse-coded
    ],
)

# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

"""
    homesf_cognitive_items(part::Symbol)

Return the Cognitive Stimulation (C) recoding rules for the given HOME Part.
"""
function homesf_cognitive_items(part::Symbol)
    return [r for r in HOMESF_SCHEMA[part] if r.scale == :C]
end

"""
    homesf_emotional_items(part::Symbol)

Return the Emotional Support (E) recoding rules for the given HOME Part.
(Saved for future use.)
"""
function homesf_emotional_items(part::Symbol)
    return [r for r in HOMESF_SCHEMA[part] if r.scale == :E]
end
