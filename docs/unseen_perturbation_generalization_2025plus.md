# 2025+ unseen perturbation generalization and unpaired-cell modeling directions

Last updated: 2026-07-08

This note records recent top-journal and top-conference directions for two
closely related problems:

1. Predicting responses to unseen perturbations, including unseen genes, drugs,
   combinations, cell types, cell lines, donors, or broader biological contexts.
2. Handling the destructive-observation problem in single-cell assays, where
   control and perturbed cells are not paired observations of the same cell.

The goal is not to build a citation list. The goal is to identify method
families, their assumptions, representative papers, and how they can support a
future TriShift-style story around reference-conditioned distributional response
prediction.

## Executive view

The strongest post-2025 opportunity is a hybrid direction:

> Context-aware perturbation priors plus distributional transport for unpaired
> unseen perturbation prediction.

This combines four ingredients:

- perturbation priors: knowledge graphs, LLM gene embeddings, drug structure,
  target graphs, MoA descriptors;
- context priors: basal expression, cell-type-specific graphs, cell-line or
  donor embeddings, set-level population encoders;
- unpaired distributional modeling: OT, flow matching, diffusion bridge,
  Schrödinger Bridge, set-to-set transport;
- robust evaluation: Systema-style perturbation-specific metrics, retrieval,
  centroid accuracy, linear and mean baselines.

This is more defensible than claiming "a larger diffusion/flow model" because
it directly addresses the three hard points reviewers now care about:

- unseen perturbation representation;
- unseen cellular context;
- unpaired control-to-perturbed population shift under systematic variation.

## Direction 1: Biological priors for unseen genetic perturbations

### Core question

How can a model predict the effect of a gene perturbation that was not observed
during training?

The key issue is that a held-out gene is not just an unseen categorical ID. The
model needs a biological coordinate for the perturbation target. Recent methods
therefore use knowledge graphs, pathway graphs, protein/gene relationships,
functional text, or LLM-derived gene embeddings.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [TxPert: using multiple knowledge graphs for prediction of transcriptomic perturbation effects](https://www.nature.com/articles/s41587-026-03113-4) | Nature Biotechnology, 2026 | Uses multiple gene/product relationship graphs, including STRING, GO and perturbation-derived graphs, to encode perturbation targets. | Strongest recent reference for unseen genetic perturbation, double perturbation and cross-cell-line OOD prediction. | Primarily predicts transcriptomic effects/aggregate profiles; not a full unpaired single-cell distribution transport method. |
| [Scouter predicts transcriptional responses to genetic perturbations with large language model embeddings](https://www.nature.com/articles/s43588-025-00912-8) | Nature Computational Science, 2025 online; volume 6, 2026 | Uses LLM-derived gene embeddings to represent perturbations. | Useful precedent for semantic perturbation priors and zero-shot gene identity. | Prior representation helps unseen genes, but does not by itself solve reference-origin or cell-state matching. |
| [Efficient Fine-Tuning of Single-Cell Foundation Models Enables Zero-Shot Molecular Perturbation Prediction](https://arxiv.org/abs/2412.13478) | arXiv 2024, revised 2025; ICLR 2025 workshop/spotlight record visible | Adds drug-conditional adapters to single-cell foundation models while training less than 1% of parameters. | Useful for adapter-style conditioning and zero-shot drug/cell-line claims. | Not a full peer-reviewed main-conference paper; claims should be treated as a tracked method, not a settled benchmark. |
| [LangPert: LLM-Driven Contextual Synthesis for Unseen Perturbation Prediction](https://openreview.net/forum?id=Tmx4o3Jg55) | ICLR 2025 workshop/preprint-style record | Uses LLM contextual synthesis for unseen perturbation prediction. | Useful as a direction for natural-language and functional-context priors. | Workshop-level evidence; needs careful benchmarking against simple baselines. |
| [Modeling Gene Expression Distributional Shifts for Unseen Genetic Perturbations](https://arxiv.org/abs/2507.02980) | arXiv, 2025 | Predicts gene-level expression histograms conditioned on perturbations and uses LLM gene embeddings for unseen perturbations. | Important because it moves from mean expression to distributional statistics for unseen genes. | Preprint; not yet a top-journal/top-conference anchor. |

### Why this direction matters

This family gives a concrete answer to "how does the model know anything about
an unseen gene?" It also helps avoid a weak zero-shot story where the model only
interpolates among seen perturbation IDs.

### Method opportunity

A strong new direction would combine:

- KG embedding for pathway and interaction prior;
- LLM or functional text embedding for gene semantics;
- perturbation-panel-aware calibration to avoid learning only shared stress
  effects;
- distributional decoder or transport module so the output is not only a mean
  vector.

For TriShift-style work, this family supports the perturbation-prior side of the
story, but it must be paired with reference-conditioned modeling to avoid
predicting a perturbation effect without specifying the starting cellular state.

## Direction 2: Cell-context and cell-type generalization

### Core question

How can a model predict perturbation responses in a new cell type, cell line,
donor, tissue state, or experimental context?

The response to the same perturbation can differ across cellular contexts. The
main modeling problem is therefore not only perturbation identity, but the
interaction between perturbation and basal cellular state.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [Predicting and interpreting cell-type-specific drug responses in the small-data regime using inductive priors](https://www.nature.com/articles/s42256-026-01202-2) | Nature Machine Intelligence, 2026 | PrePR-CT builds cell-type-specific co-expression graphs and uses GATs to predict mean and variance responses. | Strong reference for unseen cell-type and small-data drug-response generalization. | Mainly drug perturbation and mean/variance response; not a complete generative transport model. |
| [Predicting cellular responses to perturbation across diverse contexts with STATE](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2) | bioRxiv, 2025 | STATE uses a set-level state transition model and large-scale perturbed and observational single-cell data. | Very important direction for context generalization and set-aware virtual-cell modeling. | Preprint; should be cited as emerging work unless accepted later. |
| [TxPert](https://www.nature.com/articles/s41587-026-03113-4) | Nature Biotechnology, 2026 | Adds basal state encoding and perturbation graph priors for OOD tasks including cross-cell-line prediction. | Bridges unseen perturbation and cross-context prediction. | Still closer to latent transfer and aggregate effect prediction than full single-cell population transport. |
| [Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer](https://www.nature.com/articles/s42256-025-01165-w) | Nature Machine Intelligence, 2026 | XPert models drug-induced perturbation, dose-time dynamics and cell-drug interaction with a dual-branch transformer. | Strong reference for drug, dose, time and cold-cell generalization. | Less central if the target work focuses on CRISPR genetic perturbation rather than pharmacodynamics. |
| [In silico biological discovery with large perturbation models](https://www.nature.com/articles/s43588-025-00870-1) | Nature Computational Science, 2025 | LPM represents perturbation, readout and context as disentangled dimensions across heterogeneous experiments. | Good high-level foundation-model precedent for perturbation-readout-context factorization. | Broader than single-cell perturbation response prediction; fully novel symbols remain hard without embeddings. |

### Why this direction matters

Most perturbation atlases are sparse across the Cartesian product of
perturbation, cell context and readout. Generalization to a new context is often
more biologically meaningful than random held-out perturbation splits.

### Method opportunity

A strong context-generalization method should separate:

- baseline state representation;
- perturbation representation;
- context representation;
- interaction between context and perturbation;
- uncertainty or heterogeneity in the post-perturbation distribution.

For TriShift-style positioning, this points to the need for state-compatible
reference cells rather than a global control mean or a perturbation-only effect
vector.

## Direction 3: Conditional OT and distributional transport for unpaired cells

### Core question

How can a model learn the response map when the same cell cannot be observed
before and after perturbation?

Single-cell RNA-seq is destructive. Control and perturbed cells are samples from
two distributions, not paired before/after observations. Methods in this family
therefore model population-level transport rather than direct paired regression.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [Conditional Monge Gap enables generalizable single-cell perturbation modelling](https://www.nature.com/articles/s42256-026-01242-8) | Nature Machine Intelligence, 2026 | CMonge learns conditional neural OT maps for drug, dose and combination perturbations. | Strong top-journal anchor for conditional OT and unpaired single-cell perturbation modeling. | Mainly drug/dose settings; the conditioning mechanism must be adapted for genetic perturbation priors. |
| [scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](https://arxiv.org/html/2602.07103v1) | ICLR 2026 poster; arXiv 2026 | Uses distributional flow matching and MMD-style alignment for robust perturbation prediction. | Key top-conference direction for flow matching as a scalable alternative to classical OT. | 2026, but still central because it directly targets distributional perturbation response. |
| [Departures: Distributional Transport for Single-Cell Perturbation Prediction with Neural Schrödinger Bridges](https://ojs.aaai.org/index.php/AAAI/article/view/39190) | AAAI 2026 | Uses neural Schrödinger Bridges and minibatch OT-based pairing to align control and perturbed distributions. | Strong top-conference anchor for stochastic bridge modeling of unpaired perturbation data. | Needs scrutiny on whether learned pairings reflect biology or only optimize distributional fit. |
| [Doloris: Dual Conditional Diffusion Implicit Bridges with Sparsity Masking Strategy for Unpaired Single-Cell Perturbation Estimation](https://arxiv.org/abs/2506.21107) | arXiv 2025, revised 2026 | Learns control and perturbed distributions with dual conditional diffusion models sharing a latent space; adds sparsity masking. | Important emerging method for unpaired, high-dimensional sparse single-cell perturbation data. | Preprint; use as tracked method unless accepted. |
| [CellOT: Learning single-cell perturbation responses using neural optimal transport](https://www.nature.com/articles/s41592-023-01969-x) | Nature Methods, 2023 | Neural OT for unpaired treated and untreated cell states. | Older than the requested 2025+ window, but still the baseline ancestor for this family. | Often condition-specific and less directly generalizable to unseen perturbations. |
| [Mapping cells through time and space with moscot](https://www.nature.com/articles/s41586-024-08453-2) | Nature, 2025 | Scalable multi-omics OT for mapping unpaired cells across temporal and spatial domains. | Not a perturbation predictor, but a top-journal reference for atlas-scale single-cell OT alignment. | Should be cited for cell matching/alignment infrastructure, not as a perturbation response baseline. |

### Why this direction matters

This is the closest family to the "cell mismatch" problem. It directly models
the fact that pre- and post-perturbation cells are different cells. It also gives
a stronger formulation than predicting a single endpoint mean.

### Method opportunity

A competitive new method could combine:

- conditional OT/flow/bridge for unpaired population shift;
- perturbation priors for unseen genes or drugs;
- context priors for unseen cell states;
- explicit reference-origin modeling to avoid using a global control mean;
- evaluation that checks distributional heterogeneity, not only Pearson delta.

For TriShift-style positioning, the important distinction is:

> Rather than learning a direct endpoint or a perturbation-only delta, learn a
> reference-relative transition from a state-compatible control origin.

## Direction 4: Diffusion and flow models for generative perturbation landscapes

### Core question

Can generative models learn the full landscape of cellular response rather than
only a deterministic mean shift?

This family is useful when perturbation responses are heterogeneous,
time-dependent, developmental, or dose-dependent.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [Squidiff: predicting cellular development and responses to perturbations using a diffusion model](https://www.nature.com/articles/s41592-025-02877-y) | Nature Methods, published online 2025, volume 2026 | Conditional diffusion model for cellular development, gene perturbation and drug response. | Strong top-journal diffusion reference beyond OT/flow matching. | Stronger for transient landscapes and developmental settings than for gene-ID zero-shot representation. |
| [Learning cell dynamics with neural differential equations](https://www.nature.com/articles/s42256-025-01150-3) | Nature Machine Intelligence, 2025 | scDiffEq learns neural stochastic differential equations for deterministic and stochastic cell dynamics. | Useful for perturbation-as-dynamics and fate/trajectory prediction. | Not a standard endpoint Perturb-seq benchmark method. |
| [CellFlux: Simulating Cellular Morphology Changes via Flow Matching](https://arxiv.org/abs/2502.09775) | ICML 2025 | Flow matching for perturbation-induced cell morphology changes in imaging datasets. | Important top-conference example that distribution-wise flow matching can separate perturbation effects from batch artifacts. | Imaging/morphology rather than scRNA-seq expression; useful by analogy, not direct baseline. |
| [Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics](https://arxiv.org/abs/2506.22641) | arXiv, 2025 | Focuses on mode collapse and calibration in scRNA-seq perturbation modeling. | Useful warning that generative models can match mean metrics while losing population diversity. | Preprint; primarily evaluation/modeling critique. |

### Why this direction matters

Diffusion and flow methods can represent stochastic response paths and population
heterogeneity. They are attractive when the biological response is not a single
endpoint but a distribution over possible states.

### Method opportunity

The open problem is not merely using diffusion. The hard parts are:

- conditioning on unseen perturbations;
- conditioning on unseen cellular contexts;
- preserving rare response modes;
- aligning unpaired control and perturbed populations;
- passing Systema-style perturbation-specific evaluation.

## Direction 5: Drug, dose and time-aware perturbation modeling

### Core question

How can a model generalize to unseen drugs, doses, time points or drug
combinations?

This direction is especially relevant if the task includes pharmacodynamics,
chemical perturbations, or clinical transfer.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [XPert](https://www.nature.com/articles/s42256-025-01165-w) | Nature Machine Intelligence, 2026 | Dual-branch transformer with drug features, dose-time conditioning and biological network modeling. | Strongest recent top-journal reference for dose-time drug response. | Less central for CRISPR-only perturbation prediction. |
| [PrePR-CT](https://www.nature.com/articles/s42256-026-01202-2) | Nature Machine Intelligence, 2026 | Cell-type-specific graph prior for small-data chemical perturbation response. | Strong for unseen cell type plus drug response under limited data. | Not a full distributional transport method. |
| [CMonge](https://www.nature.com/articles/s42256-026-01242-8) | Nature Machine Intelligence, 2026 | Conditional OT for one or more drugs, doses and combinations. | Directly connects drug condition generalization with unpaired distributional transport. | Needs stronger unseen-gene analog if used for genetic perturbation. |
| [PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations](https://link.springer.com/article/10.1038/s44320-025-00131-3) | Molecular Systems Biology, 2025 | Generative AI model for unseen chemical and genetic perturbation response. | Useful broad example spanning chemical and genetic perturbations. | Not as strong a venue anchor as Nature/NMI/NBT/ICML, but relevant. |

### Why this direction matters

Drug response depends on dose, exposure time, target engagement and cell context.
Methods that ignore those axes may look strong on simple splits but fail in
realistic experimental design.

### Method opportunity

For a genetic-perturbation-centered project, this direction can be used as a
contrast: drug perturbation papers often model condition variables more
explicitly than genetic perturbation papers. That suggests adding richer
condition encoding for genetic perturbations, such as knockdown strength,
guide efficiency, target pathway and perturbation modality.

## Direction 6: Heterogeneity, variance and uncertainty-aware prediction

### Core question

How can a model recover responder/non-responder structure, variance and
uncertainty, rather than only mean expression?

This is important because single-cell response is often heterogeneous. A method
that predicts the correct average but collapses variability may be biologically
misleading.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [GPerturb: Gaussian process modelling of single-cell perturbation data](https://www.nature.com/articles/s41467-025-61165-7) | Nature Communications, 2025 | Gaussian process sparse perturbation regression with uncertainty over gene-level effects. | Strong statistical baseline and uncertainty-aware reference. | Not a full single-cell generative transport model. |
| [Decoding heterogeneous single-cell perturbation responses](https://www.nature.com/articles/s41556-025-01626-9) | Nature Cell Biology, 2025 | Introduces perturbation-response score for single-cell response heterogeneity and partial perturbation. | Useful for responder heterogeneity, partial knockdown and dose-to-function analysis. | Analysis/evaluation method, not a response predictor. |
| [Modeling Gene Expression Distributional Shifts for Unseen Genetic Perturbations](https://arxiv.org/abs/2507.02980) | arXiv, 2025 | Predicts histograms and higher-order statistics such as variance, skewness and kurtosis. | Directly relevant to distributional unseen perturbation prediction. | Preprint; not yet a primary top-venue anchor. |
| [Diversity by Design](https://arxiv.org/abs/2506.22641) | arXiv, 2025 | Studies mode collapse and calibrated metrics in perturbation modeling. | Useful as a warning for distributional model evaluation. | Preprint. |

### Why this direction matters

Heterogeneity-aware modeling is a natural extension of cell-mismatch modeling.
If control and perturbed populations are distributions, then recovering only the
centroid is incomplete.

### Method opportunity

Useful targets include:

- mean and variance correlation;
- Wasserstein or MMD distance;
- responder/non-responder proportion;
- Overlap@k for response genes;
- centroid accuracy under Systema-like references;
- uncertainty calibration for weak-effect perturbations.

## Direction 7: Evaluation, baselines and systematic-variation-aware benchmarking

### Core question

How do we know a method is predicting perturbation-specific biology rather than
shared systematic variation or average stress response?

This direction is essential after 2025. Several top papers show that standard
metrics can overstate progress in perturbation prediction.

### Representative papers

| Paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation](https://www.nature.com/articles/s41587-025-02777-8) | Nature Biotechnology, 2025 | Shows common metrics can be biased by systematic variation and proposes perturbation-specific references and centroid accuracy. | Mandatory evaluation reference for unseen genetic perturbation. | Evaluation framework, not a prediction model. |
| [Benchmarking algorithms for generalizable single-cell perturbation response prediction](https://www.nature.com/articles/s41592-025-02980-0) | Nature Methods, 2025/2026 | Benchmarks 27 methods across 29 datasets with 6 metrics and multiple generalization scenarios. | Strong benchmark reference for method selection and split design. | Broad benchmark; not specific to one method mechanism. |
| [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://www.nature.com/articles/s41592-025-02772-6) | Nature Methods, 2025 | Shows several deep and foundation models do not outperform simple baselines for perturbation prediction. | Mandatory baseline-warning reference. | Should not be used to claim all deep learning is useless; use as evaluation discipline. |
| [PertEval-scFM: Benchmarking Single-Cell Foundation Models for Perturbation Effect Prediction](https://proceedings.mlr.press/v267/wenteler25a.html) | ICML 2025 | Benchmarks zero-shot single-cell foundation-model embeddings and finds limited gain under distribution shift. | Strong top-conference evidence that scFM embeddings alone are insufficient. | Evaluation paper, not a new prediction model. |
| [Virtual Cell Challenge: Toward a Turing test for the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674%2825%2900675-0) | Cell, 2025 | Community challenge framing for virtual-cell perturbation prediction and context generalization. | Useful for field-level motivation and benchmark standards. | Perspective/challenge paper, not a method. |

### Why this direction matters

After Systema and the Nature Methods baseline papers, a new perturbation
prediction method must include:

- simple mean, linear, additive and nearest-context baselines;
- split definitions that distinguish unseen perturbation, unseen context and
  unseen perturbation-context pair;
- perturbation-specific metrics that reduce shared systematic effects;
- distributional metrics that check heterogeneity;
- negative or weak-effect cases, not only strong-effect successes.

## Direction 8: Large perturbation atlases and challenge-driven modeling

### Core question

Can data scale and benchmark standardization make unseen perturbation prediction
more reliable?

Recent work increasingly treats perturbation prediction as a benchmark-driven
virtual-cell problem, similar to a CASP-style challenge for gene regulation.

### Representative resources and papers

| Resource/paper | Venue/status | Main idea | Relevance | Boundary |
|---|---|---|---|---|
| [Virtual Cell Challenge: Toward a Turing test for the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674%2825%2900675-0) | Cell, 2025 | Defines a community challenge for predicting perturbation response, emphasizing context generalization. | Useful for motivating rigorous generalization tests. | Challenge framing, not a method. |
| [Virtual Cell Challenge 2025 wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up) | ARC Institute report, 2025 | Reports that models improved some capabilities but did not consistently beat naive baselines across all metrics. | Good reality check for benchmarking claims. | Non-journal source; use as context, not primary citation. |
| [Tahoe-100M: a giga-scale single-cell perturbation atlas for context-dependent gene function and cellular modeling](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1) | bioRxiv, 2025 | Large-scale perturbation atlas for context-dependent gene function and cellular modeling. | Important data-scale direction for training and evaluation. | Preprint/data resource; not itself a modeling method. |
| [Pertpy: an end-to-end framework for perturbation analysis](https://www.nature.com/articles/s41592-025-02909-7) | Nature Methods, 2025/2026 | scverse framework for perturbation analysis, metadata and perturbation distances. | Useful ecosystem/tooling reference for harmonized perturbation analysis. | Not a response prediction method. |

## Recommended reading order

### If the goal is a new method for unseen perturbation

1. TxPert
2. Scouter
3. Systema
4. Nature Methods linear-baseline paper
5. PertEval-scFM
6. Distributional-shift histogram preprint

Reason: first understand perturbation priors, then understand why standard
metrics can be misleading.

### If the goal is a new method for cell mismatch and unpaired distributions

1. CMonge
2. scDFM
3. Departures
4. Doloris
5. CellOT
6. moscot
7. Squidiff

Reason: start from conditional OT and flow/bridge formulations, then connect to
older neural OT and broader atlas-scale OT.

### If the goal is context generalization

1. PrePR-CT
2. STATE
3. TxPert cross-cell-line task
4. XPert
5. LPM
6. Virtual Cell Challenge

Reason: context generalization needs both cell-state representation and
benchmark discipline.

## Suggested research directions

### Direction A: KG/LLM-conditioned reference transport

Problem:

- unseen genetic perturbations lack learned IDs;
- control and perturbed cells are unpaired;
- global control references can mix state and systematic effects.

Method sketch:

- encode perturbation targets with KG plus LLM embeddings;
- encode basal state or target-control set with a set encoder;
- learn a reference-relative distributional transport map;
- evaluate with Systema references, retrieval, centroid accuracy and
  distributional metrics.

Closest papers:

- TxPert, Scouter, CMonge, scDFM, Departures, Systema.

Why it is promising:

- It combines unseen perturbation representation with the unpaired-cell
  formulation, which many existing methods handle separately.

### Direction B: Cell-type-specific graph transport

Problem:

- perturbation effects vary by cell type or cell line;
- cell-type-specific GRN/co-expression structure changes the response;
- paired cells are unavailable.

Method sketch:

- build cell-type-specific co-expression or regulatory graphs;
- use those graphs to condition a transport/flow model;
- predict mean, variance and heterogeneity, not only endpoint expression.

Closest papers:

- PrePR-CT, CMonge, scDFM, moscot, GPerturb.

Why it is promising:

- PrePR-CT shows the value of context-specific graph priors, but does not fully
  solve unpaired distributional transport. That gap is methodologically clear.

### Direction C: Schrödinger Bridge or flow matching with perturbation priors

Problem:

- OT maps can be too deterministic;
- diffusion/bridge paths can model stochastic transitions;
- unseen perturbations require biological priors.

Method sketch:

- use perturbation embeddings from KG/LLM/drug structure;
- learn stochastic bridges from control to perturbed distributions;
- regularize bridge paths with biology-informed priors or graph constraints.

Closest papers:

- Departures, Doloris, Squidiff, scDFM, TxPert.

Why it is promising:

- It directly targets unpaired data while preserving heterogeneity and allowing
  conditional generalization.

### Direction D: Systema-aware benchmark-first model design

Problem:

- many models look strong because of shared systematic variation;
- linear or mean baselines can be surprisingly hard to beat;
- method claims need better evaluation hooks.

Method sketch:

- design training and evaluation around perturbation-specific effects;
- include a model component that explicitly separates shared shift from
  perturbation-specific shift;
- report standard metrics and Systema-style metrics side by side.

Closest papers:

- Systema, Nature Methods linear-baseline paper, Nature Methods benchmark,
  PertEval-scFM, TxPert.

Why it is promising:

- This turns a reviewer risk into a contribution: the model is designed to avoid
  shortcutting through systematic variation.

## Paper pool by citation priority

### Primary papers to cite for a TriShift-style related work

1. TxPert, Nature Biotechnology 2026.
2. CMonge, Nature Machine Intelligence 2026.
3. scDFM, ICLR 2026.
4. Systema, Nature Biotechnology 2025.
5. Benchmarking algorithms for generalizable single-cell perturbation response
   prediction, Nature Methods 2025/2026.
6. Deep-learning-based gene perturbation effect prediction does not yet
   outperform simple linear baselines, Nature Methods 2025.
7. Scouter, Nature Computational Science 2025 online / 2026 volume.
8. PrePR-CT, Nature Machine Intelligence 2026.
9. Departures, AAAI 2026.
10. GPerturb, Nature Communications 2025.

### Secondary papers for method expansion

1. Squidiff, Nature Methods 2025/2026.
2. XPert, Nature Machine Intelligence 2026.
3. LPM, Nature Computational Science 2025.
4. scDiffEq, Nature Machine Intelligence 2025.
5. Decoding heterogeneous single-cell perturbation responses, Nature Cell
   Biology 2025.
6. Pertpy, Nature Methods 2025/2026.
7. moscot, Nature 2025.
8. PertEval-scFM, ICML 2025.

### Emerging methods to track, but cite carefully

1. STATE, bioRxiv 2025.
2. Doloris, arXiv 2025/2026.
3. Distributional histograms for unseen genetic perturbations, arXiv 2025.
4. Diversity by Design, arXiv 2025.
5. LangPert, ICLR workshop/OpenReview 2025.
6. Tahoe-100M, bioRxiv 2025.

## Reviewer-risk notes

- Do not claim "solves zero-shot perturbation prediction" unless the split
  truly holds out perturbation identities and prevents leakage through close
  perturbation-context pairs.
- Do not compare only against deep models. Include mean, linear, additive and
  nearest-context baselines.
- Do not report only Pearson delta. Include Systema-style metrics or other
  perturbation-specific metrics.
- Do not treat single-cell foundation model embeddings as sufficient evidence
  for generalization. PertEval-scFM and the Nature Methods baseline paper show
  that this claim is fragile.
- Do not call an analysis framework a prediction baseline. Pertpy and PS are
  useful but not predictors.
- Do not mix drug, genetic and cytokine perturbations without explaining which
  condition variables are available and how they are encoded.

## Compact positioning paragraph

Recent perturbation-prediction methods address out-of-distribution response
prediction from complementary angles. Prior-based models such as TxPert and
Scouter improve the representation of unseen genetic perturbations using
knowledge graphs or language-model-derived gene embeddings, whereas
context-aware methods such as PrePR-CT, XPert and STATE emphasize cell-type,
cell-line, dose-time or set-level biological context. In parallel,
distributional methods such as CMonge, scDFM and Departures formulate
single-cell perturbation prediction as unpaired population transport, reflecting
the destructive nature of single-cell profiling. Recent evaluations, including
Systema, PertEval-scFM and large Nature Methods benchmarks, show that progress
must be measured against simple baselines and perturbation-specific metrics
because standard scores can be inflated by shared systematic variation. A
TriShift-style method should therefore be positioned at the intersection of
perturbation priors, state-compatible reference construction and
distributional response recovery.
