# Literary Translation as a Content-Form Dissociation Paradigm

**Causally Implicated Content Representations in Transformer Residual Streams: A Literary Translation Paradigm**

Connor Mahon - 21CQ2 LLC — Independent AI Interpretability Research

*Working draft — manuscript in preparation*

---

## Overview

This repository contains code, data, and materials for reproducing the experiments in our working paper on using literary translation as a paradigm for probing passage-level content representations in LLM residual streams.

**Key findings:**
- Passages from the same literary work cluster together across languages in middle-layer residual streams, following a three-phase trajectory (language-dominant → content-dominant → language-reasserting)
- Activation patching at peak content layers flips content identity downstream; patching at late layers does not
- Content is built progressively: an early-layer patch amplifies through content construction to reach parity with a direct peak-layer patch
- Steering with a content direction vector overrides work identity; a random direction of identical norm does not
- Two independent translations of the same passage are 3–4 SDs more similar than arbitrary passage pairs (z=3.9), replicating across five Homer translators spanning four centuries

## Requirements

```
transformers==4.44.0
accelerate==0.33.0
bitsandbytes==0.43.3
torch>=2.4.1
numpy
huggingface_hub
```

**Hardware:** A100 80GB SXM (Llama 3.1 70B at NF4) or any GPU with 16GB+ VRAM (Mistral 7B at FP16)

**Container disk:** 400GB recommended for Llama 70B download + cache

## Repository Structure

```
├── README.md
├── requirements.txt
├── scripts/
│   ├── complete_experiment.py      # Scaled experiment + causal tests (main script)
│   ├── reviewer_tests.py           # Translator transparency: Garnett-Maude, Homer pairwise, null distribution
│   ├── thematic_experiment_v4.py   # Crossed thematic matching (isolation vs family)
│   ├── novel_experiment.py         # Original Experiments 2-3 (AK/Notes, Gogol)
│   ├── homer_experiment.py         # Experiment 1 (Homer proems)
│   ├── multipassage.py             # Experiment 4 (multi-passage + base model)
│   ├── structured_bootstrap.py     # Bootstrap null distribution
│   └── scaled_experiment_v2.py     # Hand-curated 10+10 passages (standalone)
├── figures/
│   ├── fig1_homer_layer20.png      # Homer PCA at layer 20
│   ├── fig2a_base_layer20.png      # Three-phase trajectory (Base model)
│   ├── fig2b_base_layer40.png
│   ├── fig2c_base_layer80.png
│   ├── figA1_gogol_embed.png       # Gogol heatmaps
│   ├── figA2_gogol_layer40.png
│   ├── figA3_gogol_layer80.png
│   ├── figA4_exp2_embed.png        # Experiment 2 heatmaps
│   ├── figA5_exp2_layer40.png
│   └── figA6_exp2_layer80.png
├── paper_drafts/
│   └── draft_v2.md                     # Working draft
└── corpus/
    └── passages.md                 # All passages with sources
```

## Quick Start

### Full experiment (Llama 70B, ~15 min)

```bash
pip install -r requirements.txt
huggingface-cli login --token YOUR_TOKEN
python scripts/complete_experiment.py
```

This runs:
1. Scaled experiment (10 AK + 10 Notes English + 6 Russian probes)
2. Causal Test 1: Activation patching at L10/L40/L70
3. Causal Test 2: Steering with content direction
4. Causal Test 3: Random steering control
5. L10 propagation analysis

### Translator transparency tests (Mistral 7B, ~5 min)

```bash
python scripts/reviewer_tests.py
```

No HuggingFace token needed (Mistral is ungated).

## Models

| Model | Precision | Usage |
|-------|-----------|-------|
| meta-llama/Llama-3.1-70B-Instruct | NF4 (4-bit) | All primary experiments, causal tests |
| meta-llama/Llama-3.1-70B | NF4 (4-bit) | Base model comparison (Exp 4) |
| mistralai/Mistral-7B-Instruct-v0.3 | FP16 | Cross-architecture replication, translator transparency, thematic matching |

## Corpus

All passages are public domain. Sources:

**Russian-English (Garnett):** Anna Karenina (Tolstoy), Notes from Underground (Dostoevsky), Dead Souls and The Overcoat (Gogol), Crime and Punishment and Brothers Karamazov (Dostoevsky)

**Russian-English (Maude):** Anna Karenina (alternate translator control)

**Russian-English (Dole):** Anna Karenina (third translator)

**Russian-English (Garnett):** Chekhov, "Verotchka" (short story probe)

**Russian-English (Dole):** Afanasyev, "Ivan and the Gray Wolf" (folk tale probe)

**French-English (Marx-Aveling):** Flaubert, Madame Bovary

**French-English (Samuel):** Stendhal, Le Rouge et le Noir

**Homeric (5 translators):** Chapman, Pope, Cowper, Butler, Murray × Iliad, Odyssey

**Non-literary control:** Kinder Surprise choking hazard warning (Russian, English, French)

**Bootstrap null pool:** 25 diverse public-domain passages (full listing in paper Appendix B)

## Citation

```
@article{21cq2-literary-translation-2026,
  title={Cross-Lingual Work-Level Discrimination in Transformer Residual Streams: Evidence from Literary Translation},
  author={21CQ2 LLC},
  year={2026},
  url={https://github.com/21cq2/literary-translation-paradigm}
}
```

## AI Assistance Disclosure

Research conducted with AI assistance (Claude, Anthropic) for implementation, statistical computation, script development, and draft iteration. All experimental design decisions, corpus selection, and interpretive judgments are the author's.

## License

Code: MIT License
Corpus: All passages are public domain.
