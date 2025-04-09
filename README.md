# About

This is the code to accompany manuscript titled "Ordered Item Embeddings in Sequential Recommender Systems"

We provide several model checkpoints that can be used to reproduce the evaluations.

It is located in `data/Checkpoints`, where we name the files according to `<model>_<data>_<suffix>`.

Without suffix means vanilla, with suffix 'o' means whole models and 'z' means only item embeddings is re-used.

# Evaluation

To test the performance, one can use following command

```
cd src

python eval.py ckpt_path=<path to checkpoint> data=lastfm model=sasrec

```

# Training

If one wants to train the model according to our method, first we can use the ordered synthetic data set to induce OIE in pre-training stage.

```
cd src

python train.py data=lastfm_oie model=sasrec trainer.accelerator=gpu
```

After that the checkpoints will appear in `./logs/train/` where the best checkpoint is labelled by its epoch

Then, one can use the checkpoint for the fine-tuning stage with following:

```
# for whole models (o)
python train.py data=lastfm model=sasrec trainer.accelerator=gpu ckpt_path= <path to checkpoint> oie_learning=True

# for only item embedding (z)
python train.py data=lastfm model=sasrec trainer.accelerator=gpu ckpt_path= <path to checkpoint> oie_learning=True item=True
```

# Misc

We also put the figure production in the notebooks inside src folder.

# Acknowledgment

This code is heavily based on lightning-hydra and BSARec repository. We greatly appreciate to their contributions