"""description:
    A script to make a two-columned CoNLL file from a corpus of multigec.
    The script expects the original and the reference file. Writing is done with
    redirection. It will try to create some labels from the diffs.
"""

import sys
from pathlib import Path

from Levenshtein import ratio
from syntok.tokenizer import Tokenizer

from util import parse_md

tokenizer = Tokenizer()  # multigec tokenizer


def leven(L, R, case="i"):
    if case == "i":
        a = [x.lower() for x in L]
        b = [x.lower() for x in R]
    else:
        a = L[:]
        b = R[:]

    matrix = [[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)):
        matrix[i+1][0] = i+1
    for j in range(len(b)):
        matrix[0][j+1] = j+1

    for j in range(1, len(b)+1):
        for i in range(1, len(a)+1):
            delta = 0
            if a[i-1] == b[j-1]:
                if L[i-1] != R[j-1]:
                    delta = 0.5
                else:
                    delta = 0
            else:
                delta = 1 - ratio(a[i-1], b[j-1])
            matrix[i][j] = delta + \
                min(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1])

    I = len(a)
    J = len(b)
    score = matrix[I][J]
    ops = []
    while I > 0 and J > 0:
        new_score, new_I, new_J = sorted(
            [
                (matrix[I-1][J-1], I-1, J-1),
                (matrix[I-1][J], I-1, J),
                (matrix[I][J-1], I, J-1)
            ],
            key=lambda x: x[0]
        )[0]

        if new_score == score:
            ops.insert(0, ("equal", new_I, new_J))
        else:
            if new_I != I and new_J != J:
                ops.insert(0, ("replace", new_I, new_J))
            if new_I == I and new_J != J:
                ops.insert(0, ("insert", new_I, new_J))
            if new_I != I and new_J == J:
                ops.insert(0, ("delete", new_I, new_J))

        score, I, J = new_score, new_I, new_J

    return ops


def main(orig_file, ref_file, out_file, label_scheme="fine"):
    origs = parse_md(
        Path(orig_file).read_text(encoding="utf-8"),
        unreadable="-unreadable-"
    )
    refs = parse_md(
        Path(ref_file).read_text(encoding="utf-8"),
        unreadable="-unreadable-"
    )

    orig_toks = list(tokenizer.tokenize(origs[0]["text"]))
    ref_toks = list(tokenizer.tokenize(refs[0]["text"]))

    with open(out_file, "w", encoding="utf-8") as outf:
        for orig, ref in zip(origs, refs):
            print("###", orig["id"], file=sys.stderr)

            orig_toks = list(
                str(tok).strip()
                for tok in tokenizer.tokenize(orig["text"])
            )
            ref_toks = list(
                str(tok).strip()
                for tok in tokenizer.tokenize(ref["text"])
            )
            ops = leven(orig_toks, ref_toks, case="i")

            for op, i, j in ops:
                if op == "equal":
                    try:
                        print(
                            orig_toks[i-1 if i == len(orig_toks) else i],
                            "O",
                            sep="\t",
                            file=outf,
                        )  # weird index is "intentional", do not touch
                    except IndexError:
                        print(ops, file=sys.stderr)
                        print(orig_toks, file=sys.stderr)
                        print(ref_toks, file=sys.stderr)
                        print(len(orig_toks), len(ref_toks), i, file=sys.stderr)
                        raise
                elif op == "replace":
                    L, R = orig_toks[i], ref_toks[j]

                    if label_scheme == "coarse":
                        label = "<mask>"
                    elif L.title() == R:
                        label = "case:title"
                    elif L.lower() == R:
                        label = "case:lower"
                    elif L.upper() == R:
                        label = "case:upper"
                    elif L[:-1] == R[:-1]:
                        label = f"{R[-1]}$"
                    else:
                        label = "<mask>"

                    print(orig_toks[i], label, sep="\t", file=outf)
                elif op == "insert":
                    pass
                elif op == "delete":
                    print(orig_toks[i], "-", sep="\t", file=outf)
                else:
                    raise ValueError(op)
                if orig_toks[i-1 if i == len(orig_toks) else i] in '?.!':
                    print(file=outf)
            print(file=outf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("orig_file", help="The orig file, .md")
    parser.add_argument("ref_file", help="The ref file, .md")
    parser.add_argument("out_file", help="The output file, .conll")
    parser.add_argument(
        "--label-scheme",
        choices=("coarse", "fine"),
        default="fine",
        help="coarse will only create <mask> labels for replacements,"
             " fine will try to create more labels. default: %(default)s"
    )

    args = parser.parse_args()

    main(**vars(args))
