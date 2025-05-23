# About

This is the code to accompany the manuscript titled "Ordered Item Embeddings in Sequential Recommender Systems"

We provide several model checkpoints that can be used to reproduce the evaluations.

It is located in `data/Checkpoints`, where we name the files according to `<model>_<data>_<suffix>`.

Without a suffix means vanilla, with suffix 'o' means whole models, and 'z' means only item embeddings are reused.

# Evaluation

To test the performance, one can use the following command

Note that the checkpoint name must match the data and model

```
cd src

python eval.py ckpt_path=<path to lastfm checkpoint> data=lastfm model=sasrec

```
The options for the data argument are `lastfm`, `ml`, `beauty`, `yelp`

The options for the model argument are `sasrec`, `bert4rec`, `bsarec`


# Training

If one wants to train the model according to our method, first, we can use the ordered synthetic dataset to induce OIE in the pre-training stage.

```
cd src

python train.py data=lastfm_oie model=sasrec trainer.accelerator=gpu trainer.max_epochs=10
```

After that, the checkpoints will appear in `./logs/train/<model>/<data_seq>` where you can use the `last.ckpt`

Then, one can use the checkpoint for the fine-tuning stage with the following:

```
# for whole models (o)
python train.py data=lastfm model=sasrec trainer.accelerator=gpu ckpt_path= <path to lastfm_oie checkpoint> oie_learning=True

# for only item embedding (z)
python train.py data=lastfm model=sasrec trainer.accelerator=gpu ckpt_path= <path to lastfm_oie checkpoint> oie_learning=True item=True
```

Later, the new checkpoints will appear in `./logs/train/<model>/<data>`, which one can use for the evaluation.

As like the evaluation, for the other data sets and models, one need to specify the arguments.
# Misc

We also put the figure production in the notebooks inside the `src` folder.

Some potential issues: 
- During training, there might be a problem with metric calculation; a potential cause is the location of the tensor. One can fix this by replacing `detach()` to `cpu()` at lines 50,51,78,79 of `src/utils/metrics.py`.

# Acknowledgment

This code is heavily based on lightning-hydra and the BSARec repository. We greatly appreciate their contributions
