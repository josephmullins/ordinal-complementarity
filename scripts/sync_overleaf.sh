#!/usr/bin/env bash
# ============================================================================
# Sync the current draft (first_draft.tex) into the Overleaf project folder.
#
#   Dry run (default):   bash scripts/sync_overleaf.sh
#   Actually copy:       DRY_RUN=0 bash scripts/sync_overleaf.sh
#
# What it does (all derived dynamically from first_draft.tex, so it stays
# correct as the draft changes):
#   1. Copies first_draft.tex -> Overleaf, rewriting two paths for Overleaf's
#      flat layout:  \bibliography{draft_scratch/references} -> {references}
#                    \graphicspath{...} -> {figures/}
#   2. Copies every \includegraphics figure into Overleaf/figures/ (resolving
#      each from figures_reg/ > figures/ > slides/ucla_75/figures/ — reg wins).
#   3. Appends to Overleaf's references.bib ONLY the cited entries it is missing
#      (never overwrites the coauthor bib; idempotent).
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OVL="/Users/mullinsj/Dropbox/Apps/Overleaf/Non-parametric test of complementarities"
DRY_RUN="${DRY_RUN:-1}"
cd "$REPO"
[ -d "$OVL" ] || { echo "!! Overleaf dir not found: $OVL"; exit 1; }
[ "$DRY_RUN" = "1" ] && echo ">>> DRY RUN (set DRY_RUN=0 to actually copy)" || echo ">>> LIVE SYNC"

echo; echo "== 1. main tex (first_draft.tex, with Overleaf path edits) =="
TMP="$(mktemp)"
sed -e 's|\\bibliography{draft_scratch/references}|\\bibliography{references}|' \
    -e 's|\\graphicspath{{figures_reg/}{figures/}{slides/ucla_75/figures/}}|\\graphicspath{{figures/}}|' \
    first_draft.tex > "$TMP"
grep -q '\\bibliography{references}$' "$TMP"  || { echo "  !! bib path edit did not match — check the \\bibliography line"; exit 1; }
grep -q '\\graphicspath{{figures/}}' "$TMP"   || { echo "  !! graphicspath edit did not match — check the \\graphicspath line"; exit 1; }
if [ "$DRY_RUN" = "1" ]; then echo "  [dry] cp (edited) first_draft.tex -> $OVL/first_draft.tex";
else cp "$TMP" "$OVL/first_draft.tex"; echo "  copied first_draft.tex (bib + graphicspath rewritten)"; fi
rm -f "$TMP"

echo; echo "== 2. figures -> Overleaf/figures/ =="
[ "$DRY_RUN" = "1" ] || mkdir -p "$OVL/figures"
python3 - "$REPO" "$OVL" "$DRY_RUN" <<'PY'
import re,sys,os,shutil
repo,ovl,dry=sys.argv[1],sys.argv[2],sys.argv[3]
tex=open(os.path.join(repo,"first_draft.tex")).read()
figs=[re.sub(r'.*\{([^}]+)\}$',r'\1',m.group(0)) for m in re.finditer(r'\\includegraphics(\[[^\]]*\])?\{[^}]+\}',tex)]
dirs=["figures_reg","figures","slides/ucla_75/figures"]
miss=0
for f in figs:
    src=next((os.path.join(repo,d,f) for d in dirs if os.path.isfile(os.path.join(repo,d,f))),None)
    if not src: print("  !! MISSING FIGURE:",f); miss+=1; continue
    rel=os.path.relpath(src,repo)
    if dry=="1": print(f"  [dry] {rel}  ->  figures/{f}")
    else: shutil.copy2(src,os.path.join(ovl,"figures",f)); print(f"  figures/{f}  <-  {rel}")
print(f"  ({len(figs)-miss}/{len(figs)} figures resolved)")
sys.exit(1 if miss else 0)
PY

echo; echo "== 3. bib: append cited entries missing from Overleaf/references.bib =="
python3 - "$REPO" "$OVL" "$DRY_RUN" <<'PY'
import re,sys,os
repo,ovl,dry=sys.argv[1],sys.argv[2],sys.argv[3]
tex=open(os.path.join(repo,"first_draft.tex")).read()
cited=set(k.strip() for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]+)\}',tex) for k in m.group(1).split(',') if k.strip())
ovlbib=os.path.join(ovl,"references.bib"); ovl_txt=open(ovlbib).read()
ovl_keys=set(m.group(1).strip() for m in re.finditer(r'@\w+\{\s*([^,\s]+)\s*,',ovl_txt))
repo_txt=open(os.path.join(repo,"draft_scratch/references.bib")).read()
def entry(txt,k):
    m=re.search(r'@\w+\{\s*'+re.escape(k)+r'\s*,.*?\n\}',txt,re.S); return m.group(0) if m else None
missing=sorted(cited-ovl_keys); add=[]; bad=[]
for k in missing:
    e=entry(repo_txt,k)
    (add.append(e) if e else bad.append(k))
    print(("  + "+k) if e else ("  !! "+k+"  (cited but NOT in repo bib either — add manually)"))
if not missing: print("  (Overleaf bib already covers all %d citations)"%len(cited))
if add:
    if dry=="1": print(f"  [dry] would append {len(add)} entries to references.bib")
    else:
        open(ovlbib,"a").write("\n\n% ---- appended by scripts/sync_overleaf.sh ----\n"+"\n\n".join(add)+"\n")
        print(f"  appended {len(add)} entries")
sys.exit(1 if bad else 0)
PY

echo; [ "$DRY_RUN" = "1" ] && echo ">>> DRY RUN complete — nothing was copied." || echo ">>> SYNC COMPLETE. Recompile on Overleaf (pdflatex + bibtex)."
