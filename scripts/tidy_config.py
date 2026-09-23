"""
tidy_config.py - Consolidate repeated top-level keys in swot_config.yaml.

Usage:
    python3 scripts/tidy_config.py [--config FILE]           # preview only
    python3 scripts/tidy_config.py [--config FILE] --write   # rewrite, saving a .bak copy first

Several scripts print blocks to append to the config (e.g. version_d_ids from
11b). PyYAML uses the last occurrence of a repeated key, so the settings are
correct, but the duplicates make the file hard to read. For each repeated key,
the last value is moved to the position of the first occurrence and the later
copies are deleted. The file is written only if the tidied version loads to
exactly the same settings as the original.

Inputs:  swot_config.yaml (or --config)
Outputs: the tidied config and a <config>.bak copy (with --write only)
"""

import argparse
import re
import shutil
import sys

import yaml

KEY = re.compile(r"^([A-Za-z_][\w-]*)\s*:")    # top-level key, e.g. "reaches:"
COMMENTED_ITEM = re.compile(r"^#\s*-\s")      # commented example item, e.g. '#  - "00000000001"'


def value_span(lines, i):
    """Return the index of the line after key line i and its value lines.

    Value lines are indented lines or '- ' list items. Blank lines between
    value lines are included; trailing blank lines are not.
    """
    end, j = i + 1, i + 1
    while j < len(lines):
        s = lines[j]
        if s.strip() == "":
            j += 1
            continue
        if s[0] in " \t" or s.startswith("-"):
            j += 1
            end = j
            continue
        break
    return end


def tidy(text):
    lines = text.splitlines()
    # Start and end line of every occurrence of each top-level key
    spans = {}
    for i, s in enumerate(lines):
        m = KEY.match(s)
        if m:
            spans.setdefault(m.group(1), []).append((i, value_span(lines, i)))
    repeated = {k: v for k, v in spans.items() if len(v) > 1}
    if not repeated:
        return text, []

    replace, delete = {}, set()
    for key, occ in repeated.items():
        (first_start, first_end), (last_start, last_end) = occ[0], occ[-1]
        # The template has commented example items below placeholders such as
        # "reaches: []"; delete those as well
        k = first_end
        while k < len(lines) and COMMENTED_ITEM.match(lines[k]):
            delete.add(k)
            k += 1
        replace[first_start] = (first_end, lines[last_start:last_end])
        for start, end in occ[1:]:
            delete.update(range(start, end))

    out, i = [], 0
    while i < len(lines):
        if i in replace:
            end, new = replace[i]
            out.extend(new)
            i = end
            continue
        if i not in delete:
            out.append(lines[i])
        i += 1

    # Collapse the runs of blank lines left by deleted blocks
    tidied = re.sub(r"\n{3,}", "\n\n", "\n".join(out).rstrip() + "\n")
    return tidied, sorted(repeated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the file (the original is saved as .bak)")
    args = ap.parse_args()

    text = open(args.config).read()
    new, keys = tidy(text)
    if not keys:
        print(f"{args.config}: no repeated keys; no changes needed.")
        return
    # Safety check: leave the file untouched if the loaded settings would change
    if yaml.safe_load(new) != yaml.safe_load(text):
        sys.exit("The tidied file would load differently from the original; nothing written.")

    print(f"Keys that appeared more than once: {', '.join(keys)}")
    print("Each now appears once, at its first position, with the value that was in effect.")
    print("Settings are unchanged.\n")
    if not args.write:
        print("----- tidied file (not yet written) -----")
        print(new)
        print("-----------------------------------------")
        print("If the result is correct, run again with --write.")
        return
    shutil.copy(args.config, args.config + ".bak")
    with open(args.config, "w") as f:
        f.write(new)
    print(f"Rewrote {args.config}; the original is in {args.config}.bak")


if __name__ == "__main__":
    main()
