"""
Regenerates claude/lomb_prompts_source.md's file list + concatenated
source (everything below the "Package layout" code block) from the actual
files on disk right now. Does NOT touch anything above that point (the
changelog) -- that stays hand-written per entry, prepended manually before
running this. Run from inside lomb_prompts/, then paste stdout's two
sections (package layout + concatenated source) into the project doc
under their existing headers.
"""

import pathlib

ROOT = pathlib.Path(__file__).parent
FILES = sorted(
    p for p in ROOT.rglob("*.py")
    if p.name != "build_source_doc.py" and "__pycache__" not in p.parts
) + sorted(ROOT.glob("*.json")) + sorted(ROOT.glob("*.txt"))

if __name__ == "__main__":
    rel = [p.relative_to(ROOT) for p in FILES]
    print("Package layout:\n\n```")
    for p in rel:
        print(f"./{p.as_posix()}")
    print("```\n")

    for p in rel:
        ext = "json" if p.suffix == ".json" else "python" if p.suffix == ".py" else ""
        print(f"---\n\n## `lomb_prompts/{p.as_posix()}`\n\n```{ext}")
        print((ROOT / p).read_text(encoding="utf-8").rstrip("\n"))
        print("```\n")
