"""description:
    Apply a flair error-tagger to some input markdown file. The script will output
    data into another markdown file that can be used in multigec evaluation.
"""

import re

from pathlib import Path

from flair.nn import Classifier
from flair.data import Sentence

from syntok import tokenizer

from transformers import pipeline


tok = tokenizer.Tokenizer()


def tokenize(text):
    return [str(token).strip() for token in tok.tokenize(text)]


def sentencize(text):
    tokens = [str(token).strip() for token in tok.tokenize(text)]
    sentences = []
    sentence = []

    for token in tokens:
        sentence.append(token)
        if token in ".?!":
            sentences.append(sentence[:])
            del sentence[:]

    if sentence:
        sentences.append(sentence[:])
        del sentence[:]

    return sentences


def parse_md(content, unreadable="", normalize_to="u00a7u00a7u00a7"):
    """Parse MD file, mostly copied from tools provided by multigec."""
    essays = []
    for essay in content.split("### essay_id = ")[1:]:
        essay_id, text = essay.split("\n", maxsplit=1)
        if unreadable:
            text = text.replace(unreadable, normalize_to)
        essays.append({"id": essay_id.strip(), "text": text.strip()})

    return essays


def mlm_independent(mask_filler, tokens, indices=None, threshold=0.0):
    if indices is None:
        indices = list(range(len(tokens)))
    else:
        indices = sorted(indices)

    new = tokens[:]
    for i in indices:
        data = tokens[:]
        data[i] = "<mask>"
        output = mask_filler(" ".join(data), top_k=1)[0]

        if output["score"] >= threshold:
            new[i] = output["token_str"]

    return new


def mlm_contextual(mask_filler, tokens, indices=None):
    raise NotImplementedError("mlm_contextual not implemented")


def mlm_best_ic(mask_filler, tokens, indices=None):
    raise NotImplementedError("mlm_best_ic not implemented")


strategy2func = {
    "independent": mlm_independent,
    "contextual": mlm_contextual,
    "vote-ic": mlm_best_ic,
}


def main(input_file, model_file, output_file, MLM=None, masking_strategy=None):
    classifier = Classifier.load(model_file)

    if (MLM is None) ^ (masking_strategy is None):
        raise ValueError("MLM and masking_strategy should both be set.")

    mask_filler = None
    unmasker = None
    if MLM is not None:
        mask_filler = pipeline("fill-mask", MLM, device=0)
        unmasker = strategy2func[masking_strategy]

    raw_text = Path(input_file).read_text(encoding="utf-8")
    essays = parse_md(raw_text)
    with open(output_file, "w", encoding="utf-8") as outf:
        for essay in essays:
            title = essay["id"]
            text = essay["text"]
            sentences = [Sentence(tokens) for tokens in sentencize(text)]

            classifier.predict(sentences)
            print(title)
            print("### essay_id =", title, file=outf)
            for sentence in sentences:
                tokens = [token.text for token in sentence]
                mask_indices = []
                for token in reversed(sentence):
                    idx = token.idx-1
                    tag = token.get_label().value
                    if tag == 'O':
                        pass
                    elif tag == "<mask>":
                        mask_indices.insert(0, idx)
                    elif tag == "-":
                        # TODO: CHECKME
                        masks = [idx-1 for idx in mask_indices]
                    elif tag == "case.title":
                        tokens[idx] = tokens[idx].title()
                    elif tag == "case.lower":
                        tokens[idx] = tokens[idx].lower()
                    elif tag == "case.upper":
                        tokens[idx] = tokens[idx].upper()
                    elif tag.endswith("$"):
                        last = tag[0]
                        tokens[idx] = tokens[idx][:-1] + last

                if unmasker is not None:
                    tokens = unmasker(mask_filler, tokens, mask_indices)

                to_write = " ".join(tokens)
                to_write = re.sub(r"\s*'\s*", "'", to_write)
                print(to_write, end=" ", file=outf)
            print("\n", file=outf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("input_file", help="Path to the input file, .md")
    parser.add_argument("model_file", help="Path to the flair model, .pt")
    parser.add_argument("output_file", help="Path to the output file, .md")
    parser.add_argument(
        "--MLM",
        help="Name or path to the Masked Language model to use for masked predictions."
    )
    parser.add_argument(
        "--masking-strategy",
        default=None,
        choices=sorted(strategy2func.keys()),
        help="The strategy to use for selecting the masked prediction to use."
    )

    args = parser.parse_args()

    main(
        **vars(args)
    )
