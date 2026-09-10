#!/usr/bin/env python3
"""Every markdown anchor and relative file link in this repo must resolve.

Both kinds have broken here before: section headings get rewritten while the
links to them do not, and file paths drift when scripts move. Neither shows up
until someone follows the link.
"""
import os, re, subprocess, sys

def tracked(pattern):
    out = subprocess.run(["git", "ls-files", pattern], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l]

def anchors(path):
    """GitHub's slug: lowercase, drop non-word chars, spaces to hyphens."""
    out = set()
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            out.add(re.sub(r"[^\w\s-]", "", m.group(1).strip().lower(), flags=re.UNICODE)
                      .replace(" ", "-"))
    return out

md = tracked("*.md")
anch = {f: anchors(f) for f in md}
bad = []

for f in md:
    text = open(f, encoding="utf-8").read()
    base = os.path.dirname(f)

    for a in re.findall(r"\]\(#([^)\s]+)\)", text):
        if a not in anch[f]:
            bad.append(f"{f} -> #{a}  (no such heading in this file)")

    for tgt, a in re.findall(r"\]\(([A-Za-z0-9._/-]+\.md)#([^)\s]+)\)", text):
        p = os.path.normpath(os.path.join(base, tgt))
        if p not in anch:
            bad.append(f"{f} -> {tgt}  (file not tracked)")
        elif a not in anch[p]:
            bad.append(f"{f} -> {tgt}#{a}  (no such heading there)")

    # Relative links to files in the repo, ignoring URLs and pure anchors.
    for tgt in re.findall(r"\]\((?!https?://|#|mailto:)([^)\s#]+)\)", text):
        p = os.path.normpath(os.path.join(base, tgt))
        if not os.path.exists(p):
            bad.append(f"{f} -> {tgt}  (path does not exist)")

for b in bad:
    print(f"::error::broken link: {b}")
print(f"checked {len(md)} markdown files, {len(bad)} broken links")
sys.exit(1 if bad else 0)
