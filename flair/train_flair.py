"""description:
    Train flair tagger on some data. Expected file format is two-columned CoNLL
    file where first column is tokens and second is labels.

    For multigec, the diffcorpus.py can create such corpora.
"""

# mostly taken from flair tutorial

from pathlib import Path

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


def dump(d, file):
    print("key", "value", sep="\t", file=file)
    for k in sorted(d):
        print(k, d[k], sep="\t", file=file)


def main(
    corpus_folder,
    train_file,
    model,
    dev_file=None,
    test_file=None,
    subtoken_pooling="mean",
    n_epochs=10,
    learning_rate=5.0e-6,
    batch_size=4,
    output_directory=None
):
    columns = {0: 'text', 1: 'GEC'}

    corpus: Corpus = ColumnCorpus(
        corpus_folder,
        columns,
        train_file=train_file,
        test_file=test_file,
        dev_file=dev_file,
    )

    label_type = 'GEC'
    label_dict = corpus.make_label_dictionary(
        label_type=label_type, add_unk=True)

    embeddings = TransformerWordEmbeddings(
        model=model,
        layers="-1",
        subtoken_pooling=subtoken_pooling,
        fine_tune=True,
        use_context=True,
    )

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type='GEC',
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpus)
    output_directory = output_directory or f'taggers/{model}/{Path(corpus_folder).name}'

    trainer.fine_tune(
        output_directory,
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        # remove mini_batch_chunk_size to speed up computation if you have a big GPU
        # mini_batch_chunk_size=1,
        max_epochs=n_epochs,
    )

    data_dump = {
        "corpus": corpus_folder,
        "model": model,
        "subtoken pooling": subtoken_pooling,
        "N epochs": n_epochs,
        "learning rate": learning_rate,
    }
    with open(Path(output_directory) / "train_config.csv", "w", encoding="utf-8") as output_file:
        dump(data_dump, file=output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "corpus_folder", help="The folder where train/dev/test files are.")
    parser.add_argument("train_file", help="The train file name, conll")
    parser.add_argument("model", help="The model to fine-tune")
    parser.add_argument("--dev-file", help="The dev file name, conll")
    parser.add_argument("--test-file", help="The test file name, conll")
    parser.add_argument(
        "--subtoken-pooling",
        choices=("first", "last", "first_last", "mean"),
        default="mean",
        help="The pooling strategy for subtokens"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="The number of epochs (default=%(default)s)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5.0e-6,
        help="The initial learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="The batch size. Default: %(default)s"
    )
    parser.add_argument("--output-directory", help="The output directory")

    args = parser.parse_args()

    main(**vars(args))
