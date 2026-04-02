# Causally Functional Content Representations in Transformer Residual Streams: A Literary Translation Paradigm

**21CQ2 LLC — Independent AI Interpretability Research**

---

## Abstract

Literary translations share semantic content while differing in every surface feature — a natural content-form dissociation that would be difficult to construct synthetically. Using this paradigm, we extract residual stream activations from parallel passages in Llama 3.1 70B and Mistral 7B. Passages expressing the same content cluster together at middle layers with a three-phase trajectory (language-dominant → content-dominant → language-reasserting) that replicates across six literary experiments, three genres, and four languages. The phenomenon generalizes to nine languages overall: five content classes (four UDHR articles plus a product safety warning) in six languages including Arabic and Chinese produce 27/27 LOO classification, and bespoke paragraphs in nine languages — text that cannot exist in any training corpus — produce 18/18 classification, ruling out memorization. Causal tests across four work pairs establish that these representations are not geometric artifacts: activation patching at layer 40 flips content identity downstream in all four pairs, and steering with a content direction vector overrides work identity while a random direction of identical norm does not (4–13× specificity ratio). Subspace-specific patching reveals that a single direction in the 8,192-dimensional residual stream captures 98.6% of the whole-stream patching effect — content identity is concentrated, not distributed. A propagation analysis reveals that content is built progressively — an early-layer patch amplifies through content construction to reach parity with the direct peak-layer patch by layer 50. Translator transparency (z=3.9 against 91 passage pairs, five Homer translators spanning four centuries) confirms the paradigm measures content, not style.

## 1. Introduction

Do large language models develop internal representations that correspond to passage-level meaning? If they do, passages expressing the same content in different languages should converge in the model's representational space — and manipulating these representations should change downstream processing.

We use literary translation to test this. Translations share semantic content while differing in lexicon, syntax, morphology, and script — a content-form dissociation that would be difficult to construct synthetically without introducing its own artifacts. We report three categories of findings: *correlational* (same-work passages cluster across languages, Sections 3–4), *causal* (patching and steering establish that the representations are causally functional, concentrated in a single direction, Section 5), and *methodological* (literary translation provides a natural factorial control structure for interpretability research, Section 2.2).

A caveat from translation theory: the separability of "content" and "form" in literary translation is precisely what theorists have spent decades complicating (Venuti, 1995; Meschonnic, 2007). Our paradigm sidesteps this debate by asking not "do translations preserve content?" but "what does the model treat as invariant across translations?" The finding that five Homer translators spanning verse (Pope, Cowper) and prose (Butler, Murray) cluster together suggests the model captures something that survives even transformations theorists would characterize as meaning-altering. What LLMs treat as "content" is an empirical finding, not an assumption.

## 2. Method

### 2.1 Models, Extraction, and Statistics

Llama 3.1 70B Instruct (Meta, 2024) at NF4 quantization (A100 80GB); Mistral 7B Instruct v0.3 (Jiang et al., 2023) at FP16 for cross-architecture replication. Residual stream activations at every layer, mean-pooled across tokens (special tokens excluded). Last-token pooling produces the same patterns with 2.7× larger margins (Appendix A).

For Experiments 2–5 (n=4), exact permutation is uninformative (minimum p=0.5). We use a structured bootstrap: for each of 10,000 iterations, draw 4 passages without replacement from a pool of 25 diverse public-domain passages (Appendix B), compute cosine similarity margins under all three possible pairings of 4 items into 2 pairs, and take the maximum — giving the null its best chance. This is conservative: the maximum-over-pairings inflates the null distribution. Pool sensitivity analysis (100 subsamples of 15/25, max p=0.0012) confirms robustness. Per-layer measurements are not independent; we report them as a trajectory profile, not as independent tests.

### 2.2 The Control Ladder

Literary corpora provide a natural factorial control structure:

| Control | What it isolates | Example |
|---------|-----------------|---------|
| Same translator, different works | Content from translator style | Garnett's AK vs. Garnett's Notes |
| Same author, different works | Content from authorial voice | Gogol's Dead Souls vs. Overcoat |
| Different translator, same work | Content from translator fingerprint | Garnett vs. Maude vs. Dole |
| Multiple passages, same work | Global identity from local features | AK Parts 1–8 |
| Cross-lingual, same work | Content from language surface | AK Russian vs. AK English |
| Cross-genre | Genre from content | Novels vs. folk tales vs. short stories |
| Cross-architecture | Model-general from architecture-specific | Llama 70B vs. Mistral 7B |

All passages are public domain, ~80–150 tokens of narrative prose.

## 3. Correlational Evidence

### 3.1 Confound Elimination

| Experiment | Design | Confound eliminated | Result |
|------------|--------|-------------------|--------|
| 1. Homer | 21 passages, 5 languages, 5 translators | Baseline | p=0.004 |
| 2. AK / Notes | Russian + Garnett English, n=4 | Proper nouns | p<0.001, L10–50 |
| 3. Gogol | Same author, different translators, n=4 | Authorial style | p<0.001, L10–30 |
| 4. Multi-passage + Base | 12 passages, Instruct + Base | Passage idiosyncrasy, instruction tuning | 6/6, p=0.016 |
| 5. CP / BK | Same author + translator, n=4 | Pair-specificity | p<0.001, L20–50 |

*Table 1. The three-phase trajectory replicates across all five experiments and on Mistral 7B at FP16 (30/33 layers).*

![Figure 1: Homer PCA Layer 20](fig1_homer_layer20.png)

*Figure 1. PCA of residual stream activations at layer 20 (Llama 3.1 70B, Experiment 1). Red = Iliad translations, blue = Odyssey translations (circles = English, diamonds = Greek, triangles = French/German). Five English translators cluster by epic, not by translator. French and German proems cluster with their corresponding epic, not with each other. Greek proems (right) cluster by language. PC1 (39%) captures the content axis.*

### 3.2 Scaled Experiment

Ten passages per work from AK and Notes (hand-curated, diverse chapters, mix of proper-noun-heavy and proper-noun-free). LOO classification on 20 English passages: 19/20 correct at layers 10–60 (binomial p=0.00002). Cross-lingual: 6/6 Russian probes correct at layers 25–35 and 60–75.

![Figure 2a: Layer 20](fig2a_base_layer20.png) ![Figure 2b: Layer 40](fig2b_base_layer40.png) ![Figure 2c: Layer 80](fig2c_base_layer80.png)

*Figure 2. Three-phase trajectory in PCA of residual stream activations (Llama 3.1 70B Base, Experiment 4). Red = Anna Karenina, blue = Notes from Underground; circles = English, diamonds = Russian. Left (layer 20): content clustering emerges — works separate, languages interleave within clusters. Center (layer 40): content dominance peaks — tightest separation between works. Right (layer 80): language reasserts — Russian (right) separates from English (left), content becomes secondary.*

### 3.3 Translator Transparency

Garnett vs. Maude (length-controlled, same narrative boundary): cosine similarity 0.96–0.98 at every layer (Mistral 7B), content wins 33/33. Against a null distribution of 91 same-language passage pairs: z=3.9 at layer 20, 100th percentile. Five Homer translators (Chapman 1598–1611 to Murray 1924): within-epic mean 0.90, cross-epic 0.83 at layer 20 (Llama 70B). Two translations of the same passage — different vocabulary, different syntax — are nearly indistinguishable to the model at every depth.

### 3.4 Genre and Language Diversity

| Work | Genre | Language pair | Same-work sim (L40) | Content win |
|------|-------|--------------|--------------------|----|
| Chekhov, "Verotchka" | Short story | Russian–English | 0.859 | 68/81 |
| Afanasyev, "Ivan and the Gray Wolf" | Folk tale | Russian–English | 0.912 | 69/81 |
| Flaubert, Madame Bovary | Novel | French–English | 0.918 | 74/81 |
| Stendhal, Le Rouge et le Noir | Novel | French–English | 0.907 | 76/81 |

*Table 2. Content gap +0.224 at layer 50. The trajectory replicates across genres and language pairs.*

### 3.5 Beyond Literary Prose

The literary experiments (Sections 3.1–3.4) use 19th-century European prose — a corpus constrained by public domain availability. Two additional experiments test whether the phenomenon extends to other registers and languages, and whether training data memorization could explain the results.

**UDHR: four articles, six languages.** We extract residual stream activations for four articles of the Universal Declaration of Human Rights — Article 1 (dignity and equality), Article 2 (non-discrimination), Article 5 (prohibition of torture), and Article 17 (property rights) — in English, French, Russian, German, Arabic, and Chinese, plus a Kinder Surprise choking hazard warning in three languages as a fifth content class. LOO classification: 27/27 correct at layers 20–70. The model discriminates five distinct propositions across six languages, including two non-Latin scripts (Arabic and Chinese), with perfect accuracy across the entire content-dominant phase.

The three-phase trajectory replicates: content gap emerges at layer 10 (+0.002), peaks at layer 60 (+0.117), and dissolves at layer 80 (−0.085). Arabic and Chinese passages classify correctly at every tested layer.

**Bespoke text: nine languages, zero training data.** To rule out memorization, we wrote two short paragraphs of mundane contemporary prose — one describing a poorly maintained road, one describing a child and a park — and machine-translated them into French, Russian, German, Arabic, Chinese (Simplified and Traditional), Japanese, and Uzbek. These passages cannot exist in any training corpus.

LOO classification: 18/18 correct at layers 20–30. The model correctly assigns each passage to its content class across nine languages and four script families (Latin, Cyrillic, CJK, Arabic). Content margins are thinner than for literary prose (+0.024 at peak vs. +0.224), consistent with shorter passages (~40 tokens) and less distinctive content. Uzbek passages misclassify at deeper layers, consistent with the tokenization floor observed for Ancient Greek — Llama's Uzbek tokenization inflates token counts by ~50%, degrading the content signal.

| Stimulus | Register | Languages | LOO | Memorization risk |
|----------|----------|-----------|-----|-------------------|
| Literary translation | 19th-century prose | Ru, En, Fr | 19/20 | Moderate |
| UDHR articles | Legal/declarative | En, Fr, Ru, De, Ar, Zh | 27/27 | High |
| Bespoke paragraphs | Mundane contemporary | En, Fr, Ru, De, Ar, Zh, Jp, Zht, Uz | 18/18 | Zero |

*Table 2a. The paradigm generalizes across registers and languages. The bespoke result rules out memorization: text written during the experiment, translated by machine, and never seen in training produces the same content clustering.*

## 4. Baselines

**TF-IDF.** Cross-lingual same-work similarity: 0.001. The residual stream gap is +0.144; TF-IDF's is −0.146.

**LaBSE.** LaBSE (Feng et al., 2022) produces larger content margins (gap +0.606 vs. +0.144). The paradigm does not outperform a dedicated cross-lingual model on static content detection. Its contribution is the layer-wise trajectory — revealing *how* content representations emerge, peak, and dissolve during processing, which no single-embedding model can show.

**Base model.** Larger content margins than Instruct (+0.164 vs. +0.113 at peak). The effect is a pretraining phenomenon.

## 5. Causal Evidence

### 5.1 Activation Patching

We replace one work's source-language residual stream with another work's at a target layer and measure the downstream content shift. To establish that the causal effect generalizes, we test four work pairs spanning three language configurations and three genre pairings.

**Layer specificity across four pairs.**

| Pair | Languages | L10 | L40 | L70 |
|------|-----------|-----|-----|-----|
| AK / Notes | Russian | +0.048 | **+0.248** | +0.037 |
| Bovary / Stendhal | French | +0.088 | **+0.196** | +0.133 |
| Chekhov / Afanasyev | Russian (story/folk tale) | +0.114 | **+0.124** | +0.047 |
| Bovary / Chekhov | French → Russian | +0.099 | **+0.349** | +0.083 |

*Table 3. Layer 40 produces the largest content shift in all four pairs, though the margin is attenuated for Chekhov/Afanasyev (+0.124 vs. +0.114 at L10) — consistent with the progressive construction finding that early-layer features can carry substantial content information for some work pairs. The effect generalizes across same-language (Russian, French), cross-language (French/Russian), and cross-genre (novel/folk tale, novel/short story) pairs.*

The Bovary/Chekhov result is notable: patching a French novel's residual stream with a Russian short story's activations at layer 40 produces the largest shift in the table (+0.349). The content direction operates across languages and genres simultaneously.

**Progressive construction.** Tracing the AK/Notes L10 patch downstream reveals that its effect *amplifies* through content construction:

| Patch at L10, measured at | Shift |
|--------------------------|-------|
| L20 | +0.011 |
| L30 | +0.014 |
| L40 | +0.023 |
| L50 | **+0.041** |
| L60 | **+0.043** |

*Table 3a. By L50, the L10 patch (+0.041) reaches parity with the direct L40 patch. Content is built progressively from early-layer features, not assembled at mid-network depth. (Progressive construction measured against 10-passage scaled centroids; Table 3 uses single-passage centroids, producing larger absolute shifts.)*

The "three phases" are a measurement phenomenon: early layers are already constructing content features that accumulate and become detectable in cosine space at mid-network depth. The trajectory describes when the signal becomes visible, not when it begins.

### 5.2 Steering and Random Control

We add the content direction (work A centroid minus work B centroid at layer 40) to work B's source-language representation at layer 40, and compare against a random direction of identical norm. Measured at layer 45.

| Pair | Content shift | Random shift | Ratio |
|------|-------------|-------------|-------|
| AK / Notes | +0.648 | +0.049 | 13× |
| Bovary / Stendhal | +0.679 | +0.121 | 6× |
| Chekhov / Afanasyev | +0.832 | +0.210 | 4× |
| Bovary / Chekhov | +0.747 | +0.085 | 9× |

*Table 4. Content steering shifts the representation toward the target work in all four pairs. Random steering of identical magnitude does not. The effect is directionally specific and generalizes across languages and genres.*

### 5.3 Subspace-Specific Patching

The preceding experiments replace the entire residual stream (Section 5.1) or add a direction to it (Section 5.2). Neither isolates the content-relevant subspace. To test whether content is encoded in a compact subspace or distributed across many dimensions, we compare three patching conditions at layer 40 on the AK/Notes Russian pair, measured at layer 50:

| Condition | What is replaced | Shift |
|-----------|-----------------|-------|
| Whole-stream | All 8,192 dimensions | +0.248 |
| Subspace-only | Content direction only (1 dimension) | **+0.245** |
| Orthogonal-only | All dimensions except content direction (8,191 dimensions) | +0.189 |

*Table 5. Replacing a single dimension — the content direction (AK centroid minus Notes centroid) — produces 98.6% of the whole-stream patching effect. Content identity at layer 40 is concentrated in a one-dimensional subspace.*

The subspace-only condition replaces the source passage's projection onto the content direction with the donor passage's projection onto that same direction, leaving all other dimensions unchanged. A single direction in 8,192-dimensional space captures nearly all of the content-specific causal effect. The orthogonal condition — replacing everything *except* the content direction — also shifts the representation, as expected (it changes syntax, position, register), but the content-specific shift is almost entirely accounted for by the single direction.

## 6. Boundary Conditions

**Tokenization floor.** Ancient Greek (~430 tokens via byte-level fallback) and Uzbek (~50% token inflation) fail at deeper layers. The paradigm requires competent tokenization; byte-level fallback languages are at or beyond its boundary.

**Passage length.** Literary passages (~80–150 tokens) produce content gaps of +0.224. UDHR articles (~30–100 tokens) produce +0.117. Bespoke paragraphs (~40 tokens) produce +0.024. The signal scales with passage length and content distinctiveness.

**Corpus scope.** Validated on 19th-century Russian and French prose, Homeric epic, folk tales, UDHR articles in six languages (including Arabic and Chinese), and bespoke contemporary prose in nine languages. The paradigm is not specific to literary text or European languages.

**Metric dependence.** All results use cosine similarity on mean-pooled activations. CKA or linear probes may reveal different structure.

**Quantization confound.** Llama 70B at NF4, Mistral 7B at FP16. The cross-architecture comparison confounds architecture, scale, and precision.

## 7. Related Work

Cross-lingual representation sharing has been studied in BERT-family models (Pires et al., 2019; Wu & Dredze, 2019; Conneau et al., 2020) and cross-lingual sentence embeddings (Artetxe & Schwenk, 2019). Wendler et al. (2024) establish a language-agnostic concept space in Llama 2's middle layers at the word level. Our paradigm extends this to 150-token passages and adds causal evidence. The trajectory peaks at ~40–60% of network depth in both studies.

On the interpretability side, probing classifiers (Belinkov et al., 2017) test feature decodability; sparse autoencoders (Bricken et al., 2023) decompose residual streams into interpretable features; activation patching (Meng et al., 2022) establishes causal roles. Our paradigm complements these by probing passage-level geometric organization rather than token-level features. Jermyn et al. (2025) identify shared cross-lingual features for individual concepts; our findings extend this to passage-level organization. Tang et al. (2024) identify language-specific neurons that provide a candidate mechanism for late-layer reassertion. The cross-architecture consistency is compatible with the Platonic Representation Hypothesis (Huh et al., 2024). All primary experiments use NF4 quantization (Dettmers et al., 2023).

## 8. Discussion

The content representations in the residual stream are causally functional. Patching at the content peak flips work identity in all four tested pairs; patching at late layers does not. Steering overrides work identity with 4–13× specificity over random directions. Subspace-specific patching demonstrates that a single direction captures 98.6% of the whole-stream effect — content identity at layer 40 is concentrated in a one-dimensional subspace, not distributed across the residual stream. The propagation analysis adds a mechanistic dimension: content is built progressively from early-layer features that do not yet register as content under cosine similarity. The "three phases" describe when the accumulated signal becomes detectable, not when construction begins.

The phenomenon is not limited to literary prose or European languages. UDHR articles in six languages (including Arabic and Chinese) produce perfect five-way classification. Bespoke contemporary paragraphs in nine languages — text that cannot exist in any training corpus — produce perfect two-way classification. Training data memorization cannot explain these results.

What exactly "content" consists of at the passage level — propositional structure, narrative situation, entity configuration — remains open. A crossed thematic matching experiment (Appendix D) begins to address this but is inconclusive.

**Limitations.** Three of five initial literary experiments use n=4 (scaled experiment resolves this for one work pair). The subspace patching result (Section 5.3) is demonstrated on one work pair; replication across additional pairs would strengthen the claim that content universally concentrates in a single direction. Content margins scale with passage length; the paradigm's sensitivity to very short texts (~20 tokens) is untested.

**Future directions.** Replication of subspace patching across additional work pairs and language configurations. Linear probes for work identity at late layers. Ablation of candidate language-reassertion heads (Appendix C). CKA as complementary metric. Extension to additional low-resource languages to map the tokenization boundary more precisely.

## 9. Conclusion

Using literary translation as a content-form dissociation paradigm, we demonstrate that transformer LLMs develop passage-level content representations that are causally functional — concentrated in a single direction that captures 98.6% of the whole-stream patching effect. Activation patching at peak content layers flips content identity across four work pairs; steering overrides it with directional specificity (4–13× random control). Content is built progressively from early-layer features, peaks at mid-network depth, and dissolves in late layers. The phenomenon extends beyond literary prose to legal text and mundane contemporary writing across nine languages and four script families; bespoke text absent from all training data produces the same pattern, ruling out memorization. Code, corpus, and all experimental materials are released for community use.

## References

Artetxe, M., & Schwenk, H. (2019). Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond. TACL, 7, 597–610.

Belinkov, Y., et al. (2017). What Do Neural Machine Translation Models Learn about Morphology? ACL, 861–872.

Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Anthropic Research.

Conneau, A., et al. (2020). Emerging Cross-lingual Structure in Pretrained Language Models. ACL, 6022–6034.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. NeurIPS.

Feng, F., et al. (2022). Language-agnostic BERT Sentence Embedding. ACL, 878–891.

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. ICML.

Jermyn, A., et al. (2025). On the Biology of a Large Language Model. Anthropic Research.

Jiang, A. Q., et al. (2023). Mistral 7B. arXiv:2310.06825.

Kahn-Harris, K. (2018). The Babel Message. Icon Books.

Meng, K., Bau, D., Mitchell, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS.

Meschonnic, H. (2007). Éthique et politique du traduire. Verdier.

Meta (2024). Llama 3.1 Model Card.

Pires, T., Schlinger, E., & Garrette, D. (2019). How Multilingual is Multilingual BERT? ACL, 4996–5001.

Tang, R., et al. (2024). Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models. ACL.

Venuti, L. (1995). The Translator's Invisibility: A History of Translation. Routledge.

Wendler, C., et al. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. arXiv:2402.10588.

Wu, S., & Dredze, M. (2019). Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT. EMNLP, 833–844.

---

## Appendix A: Last-Token Pooling

Last-token extraction produces the same trajectory with 2.7× larger margins. Mean pooling is retained as the primary method because it is less dependent on positional artifacts.

## Appendix B: Bootstrap Pool

25 passages spanning fiction, philosophy, science, political writing in English, Russian, French, German. Full listing: Moby Dick, Pride and Prejudice, Tale of Two Cities, Crime and Punishment (Ru/En), Communist Manifesto, Wealth of Nations, Alice in Wonderland, Metamorphosis (De/En), Republic, Principia, Walden, Brothers Karamazov (Ru), Les Misérables (Fr), Candide (Fr), Faust (De), Gettysburg Address, Don Quixote, War and Peace (Ru), Meditations, Frankenstein, Jungle Book, Eugene Onegin (Ru), Hound of the Baskervilles. Pool dominated by 19th-century Western literature.

## Appendix C: Preliminary Mechanistic Decomposition

Attention dominates content signal at 26/32 Mistral 7B layers. At layer 30, attention produces a negative content margin (−0.213); MLP near zero (+0.005). Per-head: 8/32 heads at layer 30 have negative margins (H17=−0.179, H14=−0.129, H15=−0.080). Correlational, n=4, single model. Candidates for causal validation.

## Appendix D: Thematic Sensitivity

A crossed design — two themes (isolation and family dysfunction) instantiated across four works (Anna Karenina, Brothers Karamazov, Crime and Punishment, Notes from Underground) in three languages (Russian, English, French), tested on Mistral 7B — shows thematic content detectable as a secondary signal at peak content layers when the match is genuine (Anna's despair: 20/33 layers) but not when register diverges (Ivan's dialogic crisis: 7/33). Work identity dominates overall (19/33 vs. 14/33). A Kinder Surprise choking hazard warning in three languages (cf. Kahn-Harris, 2018) confirms register discrimination (within-text cross-lingual similarity 0.922 vs. 0.713 to literary passages). Inconclusive; included for completeness.

## Appendix E: AI Assistance

Research conducted with AI assistance (Claude, Anthropic) for implementation, statistical computation, script development, and draft iteration. All experimental design decisions, corpus selection, and interpretive judgments are the author's.

## Appendix F: Code and Data

Full corpus, extraction scripts, and analysis code at github.com/21CQ2/Literary-Translation-Paradigm.
