# Related work: annotated bibliography

How each reference connects to the hypotheses of the study (H1: dropout rate vs critical sparsity,
H2: dropout rate vs weight distribution, H3: iterative vs one-shot pruning at a matched budget).

## Pruning pipelines and schedules

**Han, Pool, Tran, Dally. Learning both weights and connections for efficient neural networks. NeurIPS 2015.**
Introduced the train, prune, fine-tune pipeline with magnitude as the pruning criterion, and showed 9x to 13x
compression of AlexNet and VGG without accuracy loss. Our probe is exactly this pipeline, with the fine-tuning
stage optional so that the pruning curve measures the trained network rather than the retraining.
Connection: the magnitude criterion is what makes the per-layer weight scale matter (Section 5.6 of the report).

**Frankle and Carbin. The lottery ticket hypothesis. ICLR 2019.**
Iterative magnitude pruning plus rewinding to the initial weights finds sparse sub-networks that train to the
dense accuracy. Their iterative schedule prunes 20% of the surviving weights per round. Connection: the main
evidence that iterative pruning finds better masks than one-shot, which H3 tests at a fixed 20-epoch budget.

**Zhu and Gupta. To prune, or not to prune. arXiv 2017.**
Gradual magnitude pruning with a cubic sparsity schedule during training beats pruning once at the end for
Inception, MobileNet and LSTMs. Connection: motivates H3 and the front-loaded schedule; our result that a
front-loaded schedule only matches one-shot suggests their gain came from the long training that accompanied
the gradual schedule, not from gradualness itself.

**Blalock, Gonzalez Ortiz, Frankle, Guttag. What is the state of neural network pruning? MLSys 2020.**
Surveys 81 papers and shows results are rarely comparable because architectures, datasets, budgets and
criteria all change at once. Connection: the reason our design varies one factor (the dropout rate) and fixes
the architecture, data, optimizer, schedule and pruning criterion.

## Dropout and sparsity

**Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov. Dropout. JMLR 2014.**
Standard dropout; the intuition that a network is an ensemble of thinned sub-networks. Connection: the source
of the hypothesis that a dropout-trained network might tolerate losing weights (H1), and of the expectation that
dropout acts like a regularizer that shrinks weights (H2).

**Gomez et al. Learning sparse networks using targeted dropout. arXiv 2019.**
Dropout applied preferentially to the lowest-magnitude weights or units makes the network robust to pruning
exactly those elements, reaching high sparsity with little loss. Connection: the closest prior work; we ask
whether untargeted dropout gives a weaker version of the same effect. Our answer is no for head dropout and
the opposite for block dropout, which makes the targeting in their method essential rather than incidental.

**Molchanov, Ashukha, Vetrov. Variational dropout sparsifies deep neural networks. ICML 2017.**
Learning a per-weight dropout rate drives most rates to one, which is equivalent to removing the weight;
yields very sparse networks. Connection: shows that dropout noise and sparsity are linked when the noise is
learned per weight; our fixed, layer-wide rates do not produce sparsity.

**Bartoldson, Morcos, Barbu, Erlebacher. The generalization-stability tradeoff in neural network pruning. NeurIPS 2020.**
Argues that pruning helps generalization through the instability it introduces, like noise injection, so that
pruning and dropout play similar roles. Connection: supports treating pruning as a post-training stress test that
probes the same property dropout regularizes.

## Prunability as a measurement

**Kuhn, Lyle, Gomez, Rothfuss, Gal. Robustness to pruning predicts generalization. arXiv 2021.**
Defines prunability as the largest fraction of weights removable without raising the training loss, and finds it
predicts the generalization gap across CNNs on CIFAR-10. Connection: our summary metric and testbed. Note that
their probe is magnitude pruning without fine-tuning; our BatchNorm result (Section 5.5) applies to it directly.

## Architecture and data

**He, Zhang, Ren, Sun. Deep residual learning. CVPR 2016.** ResNet-18 in its CIFAR form is the model; its
stage structure (four stages of two BasicBlocks, 64 to 512 channels) is what the layer-wise analysis is organized by.

**Krizhevsky. Learning multiple layers of features from tiny images. 2009.** CIFAR-10.
