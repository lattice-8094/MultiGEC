import re
from pathlib import Path


def read_md(input_file):
    essays = [
        p.split("\n", 1)
        for p in Path(input_file).read_text(encoding="utf-8").strip().split("\n\n")
    ]

    for i in range(len(essays)):
        essays[i][1] = re.sub(r"([,?;.:/!])", r" \1 ", essays[i][1])
        essays[i][1] = re.sub(r"  +", r" ", essays[i][1])

    return essays


def parse_md(content, unreadable="", normalize_to="§§§"):
    """Parse MD file, mostly copied from tools provided by multigec."""
    essays = []
    for essay in content.split("### essay_id = ")[1:]:
        essay_id, text = essay.split("\n", maxsplit=1)
        if unreadable:
            text = text.replace(unreadable, normalize_to)
        essays.append({"id": essay_id.strip(), "text": text.strip()})

    return essays
