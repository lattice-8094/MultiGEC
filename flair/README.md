# Train a token classifier for multigec shared task

## Prepare data

launch diffcorpus to create data that can be used for training a token classifier.

Two options:

1. use coarse-grain tagset (3 tags : O, `<mask>` and -)
2. use fine-grain tagset : 3 previous tags, plus some specific tags according to corpus (casing, last letter replace)

Let's take Italian as an example for generating dev file from `multigec-2025-participants/` (quicker):

1. `python ./diffcorpus.py --label-scheme coarse multigec-2025-participants/italian/Merlin/it-merlin-orig-dev.md multigec-2025-participants/italian/Merlin/it-merlin-ref1-dev.md ../train_corpora/italian/dev.coarse.conll`
2. `python ./diffcorpus.py --label-scheme fine multigec-2025-participants/italian/Merlin/it-merlin-orig-dev.md multigec-2025-participants/italian/Merlin/it-merlin-ref1-dev.md ../train_corpora/italian/dev.fine.conll`

Use `-h` option to have some help. Change dev for train to have train split. You can adapt to any other language.

## Train

For now: go to flair folder and launch the train_custom.py

```
python ./train_custom.py ../../train_corpora/italian train.coarse.conll /path/to/xlm-roberta-large --dev-file dev.conll --output-directory ../../flair-models/italian/coarse/ --batch-size 8
```

Use `-h` option to have some help. You can adapt the command to other languages, coarse- / fine-grained tagset.

## Label

Use the `label.py` in flair folder to label data.

```
python ./label.py multigec-2025-participants/italian/Merlin/it-merlin-orig-dev.md ../../flair-models/italian/coarse/final-model.pt it-merlin-hypo1-dev.md --MLM /path/to/xlm-roberta-large --masking-strategy independent
```

Use `-h` option to have some help. Adapt where it is relevant.
