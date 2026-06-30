# Perturbation Prediction Manuscript Writing Patterns

This guide records the section-level writing patterns used to revise the
TriShift manuscript. It is based on the teacher's explicit guidance for the
Abstract and Introduction, plus a focused read of scPRAM, SCALE, scDFM,
Scouter, and Conditional Monge Gap for Methods, Results, and Discussion.

## Priority

1. Teacher guidance is the highest priority.
2. Bioinformatics/scPRAM organization is the main submission-format reference.
3. SCALE, scDFM, Scouter, and Conditional Monge Gap provide secondary patterns
   for mathematical setup, experiment narration, limitations, and reviewer
   risk control.
4. Generic academic polishing, including `nature-polishing`, is subordinate to
   this perturbation-prediction guide when the two conflict.

## Global Reader Logic

A perturbation-prediction paper should answer the reader's questions in this
order:

1. Why predicting perturbation responses matters.
2. Why single-cell perturbation data make the task possible but constrained.
3. Why existing methods are insufficient for the specific modeling question.
4. Why the proposed modeling form is necessary.
5. What evidence tests the modeling form.
6. Where the interpretation may fail.

For TriShift, the central modeling question is not only where a perturbation
endpoint lies, but from which control state the response starts and what
perturbation-specific shift is induced relative to that state.

## Independent Paper Summaries

These summaries preserve what each read-only paper agent found. They are not
used to override the teacher's Abstract and Introduction guidance, but they
inform Methods, Results, Discussion, figure-text coupling, notation, and
reviewer-risk control.

### scPRAM

Scope inspected: Bioinformatics paper, Methods pp. 2-4, Results pp. 4-8,
Discussion pp. 8-9.

Story chain:

- Methods: high-dimensional sparse expression requires a latent space; destructive
  sequencing creates unpaired control/perturbed observations; OT creates an
  approximate latent correspondence; attention aggregates perturbation vectors
  from similar training cells; the decoder outputs predicted perturbed
  expression.
- Results: out-of-sample accuracy; DEG recovery; cross-species heterogeneity;
  cross-individual prediction; robustness to sampling, noise, and parameters.
- Discussion: compact mechanism recap, cross-result synthesis, then limitations
  around single perturbations and missing dose/duration covariates.

Useful for TriShift:

- Bioinformatics-sized Methods organization.
- Results subsection rhythm: why the question matters, metric explanation,
  comparison, visualization or gene-level support.
- Tight main figure and supplement division.

Do not copy:

- Loose notation around dimensions or perturbation-vector symbols.
- Promotional wording such as "perfectly" or unsupported "significant
  advantage".
- Very short Discussion if TriShift still needs reference/OT/PBMC boundaries.

### SCALE

Scope inspected: Methods-like Section 3 pp. 4-9, Experiments/Results pp. 9-15,
Discussion p. 16.

Story chain:

- Methods: define control and perturbed populations, metadata and conditions;
  state that data are unpaired; formulate the task as conditional population
  transport; then introduce set-aware encoding, latent transport, condition
  injection and expression-space generation.
- Results: define benchmarks and protocol; report main results; explain metric
  conflict, especially biological metrics versus reconstruction errors; present
  efficiency and ablation; end with scaling.
- Discussion: field-level implication; evaluation reliability; shortcut/generalization
  risk; benchmark protocol mismatch; balanced closing.

Useful for TriShift:

- Start Methods with the task constraint, not the module list.
- Define whole sets/tensors before single cells or gene components.
- Explain metric conflicts rather than reporting only favorable metrics.
- Align ablation order with method components.

Do not copy:

- Non-Bioinformatics section placement.
- Draft-like transitions such as "It is obvious" or "Firstly".
- Overlong Results interpretation that belongs in Discussion.

### scDFM

Scope inspected: Methods Section 3 pp. 3-6, Experiment Section 4 pp. 6-10,
Conclusion p. 10, limitations/future work Appendix A.6 pp. 21-22.

Story chain:

- Methods: problem setup; model overview; flow-matching objective; MMD
  distribution alignment; gene expression and graph initialization; PAD
  Transformer backbone.
- Results: baseline/metric/data/split setup; additive and holdout generalization;
  drug perturbations; ablation.
- Discussion-equivalent: contribution recap, application view, explicit
  limitations around flow path, context scaling, and graph topology.

Useful for TriShift:

- Formula discipline: define symbols, give formula, explain modeling role.
- Setup section that clearly names baselines, metrics, data and splits.
- Appendix can hold detailed estimator and algorithm definitions.

Do not copy:

- Moving core limitations mostly to an appendix.
- Repetition of distribution-level fidelity claims.
- Equations in the main text that only become fully defined later in an
  appendix.

### Scouter

Scope inspected: Results pp. 3-5, Discussion p. 5, Methods pp. 5-7.

Story chain:

- Methods: reproducible task definition; inputs and outputs; prior source;
  model structure; training objective; implementation details; evaluation
  protocol.
- Results: framework explanation tied to a prior gap; aggregate benchmark
  performance; case examples; two-gene perturbation; GO-coverage failure; subtype
  and supplement deferral.
- Discussion: why the approach works; relation to foundation models and
  availability; future extension pointer.

Useful for TriShift:

- Metric orientation sentence style, for example "Both metrics favor small
  values".
- Benchmark provenance and software-version caveats.
- Figure captions can carry panel detail while Results text extracts only the
  main aggregate/case/coverage conclusions.

Do not copy:

- Symbol reuse, such as one letter for both a dataset and decoder.
- Very short Discussion without an explicit limitations paragraph.
- Strong adverbs such as "clearly" or "substantially" when metrics are mixed.

### Conditional Monge Gap

Scope inspected: Results pp. 3-8, Discussion p. 8, Methods pp. 8-11.

Story chain:

- Results: method overview; in-sample evidence that conditional information
  helps; larger dose/drug-condition settings; scaling; combinatorial
  perturbation; out-of-sample generalization; UMAP and distribution-shape
  evidence.
- Discussion: contribution and evidence breadth; mechanism and lightweight
  advantage; flexibility and combinations; limitations and future work.
- Methods: OT background; Monge/Kantorovich/Sinkhorn notation; conditional
  objective; architecture; condition encoding; evaluation protocol; datasets;
  hyperparameters; benchmarks; availability.

Useful for TriShift:

- Strongest OT notation discipline: map, measures, samples, coupling, cost
  matrix, entropy, then objective.
- Results explain weaker cases, such as high dose, embedding distance or small
  effect size, rather than hiding them.
- Discussion uses a compact four-paragraph structure with concrete limitations.

Do not copy:

- Many small Results headings if they would overload a Bioinformatics article.
- Reliance on upper/lower bound concepts without early definition.
- Dense Methods layout unless figure/notation naming is very consistent.

## Abstract

Use the teacher-provided abstract style as the direct template. Do not replace
it with patterns from other papers.

### Story Chain

Motivation:

1. State research value: predicting cellular responses matters for drug
   discovery, therapeutic design, and mechanistic interpretation.
2. State technical opportunity: single-cell perturbation sequencing can measure
   genetic, combinatorial, and environmental responses at cellular resolution.
3. State experimental constraint: profiling remains costly and incomplete.
4. State observational constraint: destructive single-cell assays prevent direct
   observation of the same cell before and after perturbation.
5. State computational gap: existing methods often predict post-perturbation
   states or condition-level averages without a state-matched control reference.
6. State risk: models may capture shared systematic variation rather than
   perturbation-specific effects.

Results:

1. Present TriShift by full name and short role.
2. State the mechanism in one sentence: denoising VAE learns comparable state
   latent space, OT identifies state-matched control references, and the model
   predicts reference-relative expression shifts using gene, protein, or
   cytokine priors.
3. State benchmark coverage: unseen genetic perturbation prediction,
   held-out-control evaluation, unseen combinatorial prediction, and
   IFN-beta-stimulated PBMC cross-cell-type response transfer.
4. State stability: limited perturbation coverage and increasing distribution
   shift, such as Dixit distance-stratified diagnostics.
5. State deeper evidence: response-associated genes, reference-centered or
   Systema-style evaluation as appropriate, and cell-level distribution
   reconstruction.
6. End with implication: better preservation of perturbation-specific signals
   and cellular heterogeneity.

### Length and Style

- Motivation: usually 90-130 English words.
- Results: usually 130-180 English words.
- Use structured abstract labels when the journal format asks for them.
- Avoid numerical overload. One compact stability or benchmark phrase is enough
  unless the abstract specifically needs a quantitative anchor.
- Use restrained verbs: "predicts", "learns", "identifies", "estimates",
  "achieves", "remains stable", "supports".
- Avoid promotional verbs: "revolutionizes", "breakthrough", "unprecedented",
  "remarkable", "perfectly", "superior".

## Introduction

Use the teacher's five-paragraph causal funnel. Do not treat the Introduction
as a list of papers.

### Five-Paragraph Story Chain

1. Background and value: perturbation responses matter for gene function,
   regulatory mechanisms, drug action, therapeutic design, and personalized
   intervention.
2. Experimental bottleneck: candidate conditions grow with genes, combinations,
   drug/cytokine treatments, cell types, and cell states; experiments are
   costly and destructive, so there is no true cell-wise before/after pair.
3. First method family: endpoint generation and disentangled representation
   methods such as scGen and BioLORD model post-perturbation states, but are
   limited when extrapolating beyond observed states or perturbation
   backgrounds. If the prediction target is compressed into one endpoint or a
   condition mean, state-dependent response differences can be weakened.
4. Second and third method families: external perturbation-prior methods improve
   unseen-perturbation generalization but perturbation identity alone does not
   determine the starting control state. OT and distributional methods such as
   CellOT and scPRAM motivate soft state correspondence under destructive,
   unpaired observations, but should not be framed as true cell-wise pairing and
   may remain limited for unseen perturbation types.
5. Motivation and TriShift: the remaining problem is to jointly specify a
   state-matched control reference and estimate the perturbation-induced shift
   relative to that reference. Then introduce TriShift, its mechanism, and the
   evidence preview.

### Length and Style

- Five paragraphs are the default.
- Each paragraph should usually be 120-220 English words.
- Each paragraph needs a clear topic sentence and a clear handoff to the next.
- Use "therefore", "however", "by contrast", "this leaves", and "for this
  reason" to make the causal chain explicit.
- Avoid literal translation and "under external perturbations..." opening if it
  reads like a definition rather than a value statement.

## Methods

Methods should make the modeling necessity obvious before naming modules. The
best reference patterns are SCALE for task formulation, CMonge for OT notation,
scDFM for formula-after-definition discipline, Scouter for reproducibility
provenance, and scPRAM for Bioinformatics section scale.

### Methods Story Chain

1. Start from the data constraint: single-cell perturbation data are destructive
   and unpaired, so the model should not imply true cell-wise trajectories.
2. Define the prediction object: control cells, perturbed cells, perturbation
   conditions, external priors, reference origins, and prediction targets.
3. State the modeling form: reference-conditioned state transition, not direct
   endpoint regression.
4. Explain the workflow as a consequence of the modeling form:
   representation learning creates a comparable state space; OT retrieves
   state-compatible reference origins; shift prediction combines reference
   state and perturbation prior; generation outputs a reference-relative
   expression shift.
5. Give mathematical details module by module.
6. Separate training-time reference construction from evaluation-time reference
   use, especially for held-out-control and PBMC target-domain-control settings.
7. Move metric implementation details, extensive hyperparameters, and auxiliary
   definitions to the Supplement unless the main text needs them to understand
   the method.

### Methods Paragraph Types

Use these paragraph types deliberately.

| Paragraph type | Job | Typical length | Reviewer risk if missing |
|---|---|---:|---|
| Problem framing | Convert destructive, unpaired observations into a modeling requirement | 80-150 words | Reviewer thinks modules are arbitrary |
| Notation setup | Define spaces, sets, vectors, indices, dimensions, labels, priors | 120-250 words | Symbol ambiguity; vector/component confusion |
| Overview | Explain the method chain and point to subsections/figure | 180-350 words | Reader cannot map equations to model |
| Module rationale | State why a module is needed before the formula | 70-140 words | Module-list prose |
| Formula block | Give objective, mapping, or loss after all symbols are introduced | 1-3 equations plus 80-180 words | Undefined variables; unreferenced equations |
| Formula interpretation | Translate symbols into model behavior and assumptions | 50-120 words | Formula appears decorative |
| Protocol/provenance | Define split, baselines, metric computation, code/data provenance | 120-250 words | Reproducibility questions |
| Boundary note | State what the procedure does not imply | 60-120 words | OT mistaken for true biological pairing |

### Notation Rules

These rules address issues reviewers and the teacher are likely to notice.

1. Define full vectors before scalar components.
   - Good: `For any training vector s_j=(s_{j,1},...,s_{j,G}) in R^G, its
     reconstruction is \\hat{s}^{rec}_j=(\\hat{s}^{rec}_{j,1},...,\\hat{s}^{rec}_{j,G}).`
   - Avoid using `x_j` as both a generic training vector and a control-cell
     symbol if `x_i^0` already denotes control cells.
2. Use separate notation for biological states and generic VAE inputs.
   - Control cells: `x_i^0`.
   - Perturbed cells: `y_j`.
   - Generic training expression vector: `s_j` or another neutral symbol.
3. Avoid `z_mu`, because a subscripted `mu` can be misread as an index.
   Prefer `mu_phi(s)`, `r_i`, `h_i`, or a clearly named latent representation.
4. Define distributions, measures, matrices, and coupling before OT formulas.
   CMonge's sequence is useful: map, measures, samples, Dirac delta, coupling
   set, cost matrix, entropy, objective.
5. Do not reuse the same symbol for different objects, such as dataset and
   decoder/generator.
6. Every numbered equation should be referenced in the text. Equations that are
   not referenced should be unnumbered.
7. Formula explanations should include dimensions when ambiguity is possible.
8. Use the same notation in the main text and Supplement. If the Supplement has
   a fuller estimator, the main text must not imply a different definition.

### Methods Language Style

Preferred:

- "The representation module is trained by minimizing..."
- "Let ... denote ..."
- "This construction allows ..."
- "The coupling is used to construct candidate references, not to infer true
  cell-wise biological pairs."
- "At evaluation time, the model uses only held-out control references as
  prediction origins."

Avoid:

- "Module 1 optimizes..."
- "This is used to..."
- "Good matching", "bad prediction", "perfect overlap".
- Long semicolon chains that list four or more facts in one sentence.
- "For simplicity" when a simplification changes interpretation. State the
  assumption directly.

### Reference Paper Lessons for Methods

- scPRAM: useful Bioinformatics-sized method chain, but its dimensions and some
  symbols are not strict enough. TriShift should borrow the VAE/OT/attention
  flow structure, not its loose notation.
- SCALE: strongest formulation pattern. It defines whole tensors/sets first,
  then samples and components. It frames the task as conditional population
  transport rather than module assembly.
- scDFM: useful "definition -> formula -> interpretation" discipline. Its
  appendix handles estimator and algorithm details.
- Scouter: useful reproducibility chain: input, prior source, architecture,
  loss, implementation, training, splits, benchmarks, metrics, resources.
- CMonge: strongest OT notation discipline and strongest separation between
  mathematical objective and engineering details.

## Results

Results should be an evidence sequence, not a panel inventory. The strongest
patterns come from scPRAM for Bioinformatics rhythm, SCALE for metric-conflict
explanation, scDFM for setup/metric/data/split clarity, Scouter for metric
orientation and case studies, and CMonge for explaining weak or anomalous
conditions rather than hiding them.

### Results Overview Story Chain

The Results opening should usually be 1-2 paragraphs and 180-300 English words.
It should not contain detailed numbers.

1. Reconnect to the central question from the Introduction and Methods.
2. State the evidence sequence in the order the reader will see it.
3. Explain how main figures and supplementary figures divide core evidence from
   diagnostics.
4. Avoid "if the method works, then..." checklists. Write a roadmap for evidence,
   not a proof plan.

### Results Subsection Story Chain

Each Results subsection should follow this order unless there is a strong reason
not to:

1. Why this evaluated question matters.
2. Exact split, held-out setting, benchmark, or case selection.
3. Metric meaning and direction when introduced for the first time.
4. Main comparison against the strongest or most relevant baseline.
5. One or two quantitative anchors tied to dataset, metric, baseline, and
   setting.
6. Concise model-behavior interpretation.
7. Boundary or supplement pointer only when needed.

### Results Paragraph Types

| Paragraph type | Job | Typical length | Required content |
|---|---|---:|---|
| Evidence roadmap | Tell reader what the Results will test | 180-300 words | empirical questions, figure order, supplement role |
| Experiment setup | Explain why and how a setting tests the claim | 100-180 words | dataset, split, held-out objects, leakage boundary |
| Metric orientation | Tell reader what a metric means and which direction is better | 60-130 words | metric, direction, reason for use |
| Main comparison | Report observed trend and strongest baseline comparison | 120-220 words | 1-2 numbers, dataset/metric/baseline |
| Mechanism interpretation | Connect result to reference-conditioned shift behavior | 60-140 words | behavior explanation, no broad speculation |
| Ablation paragraph | Map component change to observed effect | 120-220 words | variant, metric, change, implication |
| Case/visualization | Show concrete gene/cell/condition behavior | 120-200 words | why case chosen, what pattern is visible, metric consistency |
| Diagnostic/robustness | Explain performance under data scarcity, distance, noise, or shift | 150-250 words | varied factor, trend, boundary |

### Results Language Style

Preferred transition verbs and phrases:

- "To test this setting..."
- "We first evaluated..."
- "We next asked..."
- "This metric favors lower values..."
- "In contrast..."
- "This pattern indicates..."
- "The result supports..."
- "The trend is consistent with..."
- "The case study provides a gene-level view of..."

Use sparingly:

- "Notably" and "Importantly". These should mark a real interpretation, not
  routine figure reading.

Avoid:

- "Clearly", "perfectly", "significantly" without a statistical test,
  "substantially" without scale, and "superior" as a generic claim.
- Prose that says only "Fig. X shows..." without explaining what it shows.
- Listing every panel in order when a figure caption can do that.
- Hiding weak splits or anomalous conditions. Explain whether the issue is
  effect size, support distance, reference mismatch, domain shift, metric
  sensitivity, or marker sparsity.

### Reference Paper Lessons for Results

- scPRAM provides the strongest Bioinformatics rhythm: each subsection opens
  with a testable question, then defines metrics, reports comparisons and ends
  with a visual or biological interpretation.
- SCALE shows how to handle metric conflict. If a metric appears worse while a
  biologically meaningful metric improves, explain what each metric measures
  instead of selecting only the favorable one.
- scDFM shows how to make experiment setup explicit before result claims:
  baselines, metrics, data and splits are part of the evidence, not administrative
  detail.
- Scouter shows how to orient readers with metric direction and how to combine
  aggregate benchmarks with fixed cases.
- CMonge shows how to explain weak or anomalous conditions and how to pair
  distributional metrics with UMAP or pathway-level diagnostics.

For TriShift, combine these patterns into one subsection rule: evaluated
question, exact setting, metric direction, strongest baseline, sparse numerical
anchor, model-behavior interpretation and boundary.

### Reviewer-Facing Results Checks

Before accepting a Results revision, check:

1. Does every subsection answer a reviewer question, not just show a figure?
2. Are held-out-control genetic tasks separated from PBMC target-domain-control
   transfer?
3. Are metric directions defined at first use?
4. Are numerical anchors sparse but sufficient?
5. Are baselines compared fairly and by strongest relevant model, not by the
   weakest baseline?
6. Are Systema-style, reference-centered, centroid-centered, and distributional
   metrics named exactly as defined?
7. Are UMAP and violin plots framed as visualization support rather than proof?
8. Are main claims supported in the main figure, with diagnostics in the
   supplement?
9. Are unexpected or weaker results acknowledged and explained?
10. Does the paragraph close with model behavior, not broad Discussion-level
    claims?

## Discussion

Discussion should interpret the evidence and define boundaries. It should not
repeat Results subsection by subsection.

### Discussion Story Chain

Use a four- or five-paragraph structure by default.

1. Contribution and evidence synthesis: restate TriShift as a
   reference-conditioned state-transition model and summarize what the combined
   evidence supports.
2. Mechanistic interpretation: explain why state-matched references, OT
   retrieval, and external perturbation priors are complementary.
3. Relationship to prior methods: compare against endpoint generation,
   perturbation-prior methods, OT/distributional methods, and foundation models
   fairly.
4. Boundary conditions: state where the current evidence does not yet extend.
5. Future directions: give specific extensions that follow from the limitations.

### Length and Style

- Bioinformatics-style Discussion can be 500-900 English words.
- If the paper is page-constrained, preserve at least one concrete limitations
  paragraph.
- Each paragraph should usually be 90-180 words.
- Use restrained interpretation verbs: "suggests", "supports", "is consistent
  with", "may reflect", "may be limited by".
- Avoid using Discussion to introduce unsupported new claims or new results.

### Required TriShift Boundaries

At least one Discussion paragraph should cover:

1. OT references are prediction origins, not true one-to-one cell pairs.
2. Reference quality and control-pool coverage can limit prediction.
3. Held-out-control genetic tasks and PBMC target-domain-control transfer are
   related but not identical settings.
4. Protein/cytokine prior conclusions should not be generalized beyond tested
   IFNB1/PBMC evidence without more experiments.
5. Higher-order combinations, dose, time, batch effects, new cell types, and
   stronger out-of-domain conditions remain future tests unless directly
   evaluated.
6. Metrics can favor different behavior; distributional fidelity, response-gene
   recovery, and endpoint errors should be interpreted together.

### Reference Paper Lessons for Discussion

- scPRAM: short Bioinformatics-style synthesis plus explicit limitation around
  single perturbation and missing dose/duration. Useful for concise structure,
  but too brief for TriShift's reference/OT boundaries.
- SCALE: strong reviewer-facing limits around evaluator sensitivity, shortcut
  solutions, and protocol mismatch.
- scDFM: useful appendix-style future-work bullets, but TriShift should keep key
  limitations in the main Discussion.
- Scouter: useful comparison with foundation models, but too short for a full
  Bioinformatics manuscript.
- CMonge: best four-paragraph pattern: contribution, mechanism/advantage,
  flexibility, and limitations/future work.

## Figure Captions and Figure-Text Coupling

Captions should carry panel definitions; Results text should carry scientific
interpretation.

Caption requirements:

1. First sentence states the figure's main finding or role.
2. Panel labels define what each panel shows.
3. Captions state metric direction if the panel is not self-evident.
4. Captions distinguish main analysis from diagnostic or supplementary detail.
5. Captions avoid vague "good/bad" visual descriptions.

Text requirements:

1. Do not repeat every panel detail.
2. Explain the one or two key observations that support the section claim.
3. For UMAP, violin, or case panels, explain why the case is representative or
   why it is a fixed illustrative case.

## Reviewer Risk Checklist

Run this checklist before delivery.

### Macro Logic

- Does the paper move from value to constraint to gap to method to evidence?
- Does each Results subsection test a claim introduced earlier?
- Does Discussion interpret the same claims tested in Results?

### Methods Rigor

- Are all symbols defined before use?
- Are vector and scalar components distinguished?
- Are generic training vectors distinct from control and perturbed cells?
- Are distributions, coupling matrices, cost matrices, entropy terms, and loss
  terms defined with dimensions when needed?
- Are training-time and evaluation-time reference strategies separated?
- Are numbered equations referenced?

### Results Rigor

- Are metrics defined and directions stated?
- Are comparisons made against the strongest relevant baseline?
- Are quantitative anchors traceable to figures/tables/results?
- Are weak or anomalous results explained?
- Is visualization language restrained?

### Discussion Rigor

- Are prior methods credited fairly?
- Are limitations concrete and tied to experiments?
- Are claims about cytokine/protein priors limited to tested evidence?
- Are future directions specific enough to be useful?

### Style

- No oral phrasing such as "we just", "this thing", "good/bad", or "looks like".
- No hard semicolon chains.
- No em-dash connectors.
- No AI-tone promotional vocabulary.
- No unsupported "significant" language.

## Workflow for Future Manuscript Optimization

1. Load the perturbation-prediction writing skill first.
2. Use this guide as the primary style and structure reference.
3. Use `nature-polishing` only after the section job and perturbation-specific
   story chain are correct.
4. Edit Chinese manuscript first, English manuscript second, Supplement third.
5. Run an independent `pre-submission-reviewer` check after every manuscript or
   supplement optimization.
6. Fix all CRITICAL and MAJOR findings or record why they are deferred.
7. Record the review and verification in `.byte-os/STATUS.md` and the active
   Byte OS note.
