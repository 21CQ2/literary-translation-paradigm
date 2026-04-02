"""
Homer Proem Residual Stream Experiment
======================================
Cross-lingual semantic representation in LLM residual streams.

Design: 2 epics (Iliad/Odyssey) × 2 languages (Greek/English) × 5 translators
        = 14 passages total (2 Greek + 10 English)

Extracts mean-pooled residual stream activations at every layer,
then runs PCA and LOO classification on the content dimension.

Target: Llama 3.1 70B Instruct, 4-bit, 1x A100 80GB SXM
Pinned: transformers==4.44.0, accelerate==0.33.0, torch==2.4.1, bitsandbytes==0.43.3
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────────

PASSAGES = [
    # === GREEK SOURCE TEXTS ===
    {
        "id": "greek_iliad",
        "epic": "iliad",
        "language": "greek",
        "translator": "homer",
        "verse_form": "hexameter",
        "text": (
            "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος "
            "οὐλομένην, ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε, "
            "πολλὰς δ᾽ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν "
            "ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν "
            "οἰωνοῖσί τε πᾶσι, Διὸς δ᾽ ἐτελείετο βουλή, "
            "ἐξ οὗ δὴ τὰ πρῶτα διαστήτην ἐρίσαντε "
            "Ἀτρεΐδης τε ἄναξ ἀνδρῶν καὶ δῖος Ἀχιλλεύς. "
            "τίς τ᾽ ἄρ σφωε θεῶν ἔριδι ξυνέηκε μάχεσθαι; "
            "Λητοῦς καὶ Διὸς υἱός: ὃ γὰρ βασιλῆϊ χολωθεὶς "
            "νοῦσον ἀνὰ στρατὸν ὄρσε κακήν, ὀλέκοντο δὲ λαοί, "
            "οὕνεκα τὸν Χρύσην ἠτίμασεν ἀρητῆρα "
            "Ἀτρεΐδης: ὃ γὰρ ἦλθε θοὰς ἐπὶ νῆας Ἀχαιῶν "
            "λυσόμενός τε θύγατρα φέρων τ᾽ ἀπερείσι᾽ ἄποινα, "
            "στέμματ᾽ ἔχων ἐν χερσὶν ἑκηβόλου Ἀπόλλωνος "
            "χρυσέῳ ἀνὰ σκήπτρῳ, καὶ λίσσετο πάντας Ἀχαιούς, "
            "Ἀτρεΐδα δὲ μάλιστα δύω, κοσμήτορε λαῶν"
        ),
    },
    {
        "id": "greek_odyssey",
        "epic": "odyssey",
        "language": "greek",
        "translator": "homer",
        "verse_form": "hexameter",
        "text": (
            "ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον, ὃς μάλα πολλὰ "
            "πλάγχθη, ἐπεὶ Τροίης ἱερὸν πτολίεθρον ἔπερσεν: "
            "πολλῶν δ᾽ ἀνθρώπων ἴδεν ἄστεα καὶ νόον ἔγνω, "
            "πολλὰ δ᾽ ὅ γ᾽ ἐν πόντῳ πάθεν ἄλγεα ὃν κατὰ θυμόν, "
            "ἀρνύμενος ἥν τε ψυχὴν καὶ νόστον ἑταίρων. "
            "ἀλλ᾽ οὐδ᾽ ὣς ἑτάρους ἐρρύσατο, ἱέμενός περ: "
            "αὐτῶν γὰρ σφετέρῃσιν ἀτασθαλίῃσιν ὄλοντο, "
            "νήπιοι, οἳ κατὰ βοῦς Ὑπερίονος Ἠελίοιο "
            "ἤσθιον: αὐτὰρ ὁ τοῖσιν ἀφείλετο νόστιμον ἦμαρ. "
            "τῶν ἁμόθεν γε, θεά, θύγατερ Διός, εἰπὲ καὶ ἡμῖν."
        ),
    },

    # === ENGLISH ILIAD PROEMS ===
    {
        "id": "chapman_iliad",
        "epic": "iliad",
        "language": "english",
        "translator": "chapman",
        "verse_form": "fourteeners",
        "text": (
            "Achilles' baneful wrath resound, O Goddess, that imposed "
            "Infinite sorrows on the Greeks, and many brave souls losed "
            "From breasts heroic; sent them far to that invisible cave "
            "That no light comforts; and their limbs to dogs and vultures gave: "
            "To all which Jove's will gave effect; from whom first strife begun "
            "Betwixt Atrides, king of men, and Thetis' godlike son. "
            "What god gave Eris their command, and oped that fighting vein? "
            "Jove's and Latona's son; who, fired against the king of men "
            "For contumely shown his priest, infectious sickness sent "
            "To plague the army, and to death by troops the soldiers went. "
            "Occasioned thus: Chryses, the priest, came to the fleet to buy, "
            "For presents of unvalued price, his daughter's liberty; "
            "The golden sceptre and the crown of Phoebus in his hands "
            "Proposing; and made suit to all, but most to the commands "
            "Of both the Atrides, who most ruled."
        ),
    },
    {
        "id": "pope_iliad",
        "epic": "iliad",
        "language": "english",
        "translator": "pope",
        "verse_form": "heroic_couplets",
        "text": (
            "Achilles' wrath, to Greece the direful spring "
            "Of woes unnumber'd, heavenly goddess, sing! "
            "That wrath which hurl'd to Pluto's gloomy reign "
            "The souls of mighty chiefs untimely slain; "
            "Whose limbs unburied on the naked shore, "
            "Devouring dogs and hungry vultures tore. "
            "Since great Achilles and Atrides strove, "
            "Such was the sovereign doom, and such the will of Jove! "
            "Declare, O Muse! in what ill-fated hour "
            "Sprung the fierce strife, from what offended power "
            "Latona's son a dire contagion spread, "
            "And heap'd the camp with mountains of the dead; "
            "The king of men his reverent priest defied, "
            "And for the king's offence the people died. "
            "For Chryses sought with costly gifts to gain "
            "His captive daughter from the victor's chain. "
            "Suppliant the venerable father stands, "
            "Apollo's awful ensigns grace his hands: "
            "By these he begs; and lowly bending down, "
            "Extends the sceptre and the laurel crown. "
            "He sued to all, but chief implored for grace "
            "The brother-kings, of Atreus' royal race."
        ),
    },
    {
        "id": "cowper_iliad",
        "epic": "iliad",
        "language": "english",
        "translator": "cowper",
        "verse_form": "blank_verse",
        "text": (
            "Achilles sing, O Goddess! Peleus' son; "
            "His wrath pernicious, who ten thousand woes "
            "Caused to Achaia's host, sent many a soul "
            "Illustrious into Ades premature, "
            "And Heroes gave (so stood the will of Jove) "
            "To dogs and to all ravening fowls a prey, "
            "When fierce dispute had separated once "
            "The noble Chief Achilles from the son "
            "Of Atreus, Agamemnon, King of men. "
            "Who them to strife impelled? What power divine? "
            "Latona's son and Jove's. For he, incensed "
            "Against the king, a foul contagion raised "
            "In all the host, and multitudes destroyed, "
            "For that the king had with dishonour marked "
            "His priest Chryses. To the fleet he came "
            "Charged with large ransom, in his hands he bore "
            "The sacred fillets and the golden sceptre "
            "Of the shaft-armed God, and he besought "
            "The whole Achaian host, but above all "
            "The sons of Atreus, highest in command."
        ),
    },
    {
        "id": "butler_iliad",
        "epic": "iliad",
        "language": "english",
        "translator": "butler",
        "verse_form": "prose",
        "text": (
            "Sing, O goddess, the anger of Achilles son of Peleus, that brought "
            "countless ills upon the Achaeans. Many a brave soul did it send "
            "hurrying down to Hades, and many a hero did it yield a prey to dogs "
            "and vultures, for so were the counsels of Jove fulfilled from the day "
            "on which the son of Atreus, king of men, and great Achilles, first "
            "fell out with one another. And which of the gods was it that set them "
            "on to quarrel? It was the son of Jove and Leto; for he was angry with "
            "the king and sent a pestilence upon the host to plague the people, "
            "because the son of Atreus had dishonoured Chryses his priest. Now "
            "Chryses had come to the ships of the Achaeans to free his daughter, "
            "and had brought with him a great ransom: moreover he bore in his hand "
            "the sceptre of Apollo wreathed with a suppliant's wreath and he "
            "besought the Achaeans, but most of all the two sons of Atreus, "
            "who were their chiefs."
        ),
    },
    {
        "id": "murray_iliad",
        "epic": "iliad",
        "language": "english",
        "translator": "murray",
        "verse_form": "prose",
        "text": (
            "The wrath sing, goddess, of Peleus' son, Achilles, that destructive "
            "wrath which brought countless woes upon the Achaeans, and sent forth "
            "to Hades many valiant souls of heroes, and made them themselves spoil "
            "for dogs and every bird; thus the plan of Zeus came to fulfillment, "
            "from the time when first they parted in strife Atreus' son, king of "
            "men, and brilliant Achilles. Who then of the gods was it that set "
            "them to contend? The son of Leto and Zeus; for he in anger against "
            "the king roused throughout the host an evil pestilence, and the "
            "people were perishing, because the son of Atreus had dishonoured "
            "Chryses the priest. For he had come to the swift ships of the "
            "Achaeans to free his daughter, bearing ransom past counting, and in "
            "his hands the fillets of Apollo who strikes from afar, on a staff "
            "of gold; and he implored all the Achaeans, but most of all the two "
            "sons of Atreus, marshallers of the host."
        ),
    },

    # === ENGLISH ODYSSEY PROEMS ===
    {
        "id": "chapman_odyssey",
        "epic": "odyssey",
        "language": "english",
        "translator": "chapman",
        "verse_form": "heroic_couplets",
        "text": (
            "The man, O Muse, inform, that many a way "
            "Wound with his wisdom to his wished stay; "
            "That wandered wondrous far, when he the town "
            "Of sacred Troy had sack'd and shivered down; "
            "The cities of a world of nations, "
            "With all their manners, minds, and fashions, "
            "He saw and knew; at sea felt many woes, "
            "Much care sustained, to save from overthrows "
            "Himself and friends in their retreat for home; "
            "But so their fates he could not overcome, "
            "Though much he thirsted it. O men unwise, "
            "They perished by their own impieties, "
            "That in their hunger's rapine would not shun "
            "The oxen of the lofty-going Sun, "
            "Who therefore from their eyes the day bereft "
            "Of safe return. These acts, in some part left, "
            "Tell us, as others, deified Seed of Jove."
        ),
    },
    {
        "id": "pope_odyssey",
        "epic": "odyssey",
        "language": "english",
        "translator": "pope",
        "verse_form": "heroic_couplets",
        "text": (
            "The man for wisdom's various arts renown'd, "
            "Long exercised in woes, O Muse! resound; "
            "Who, when his arms had wrought the destined fall "
            "Of sacred Troy, and razed her heaven-built wall, "
            "Wandering from clime to clime, observant stray'd, "
            "Their manners noted, and their states survey'd, "
            "On stormy seas unnumber'd toils he bore, "
            "Safe with his friends to gain his natal shore: "
            "Vain toils! their impious folly dared to prey "
            "On herds devoted to the god of day; "
            "The god vindictive doom'd them never more "
            "(Ah, men unbless'd!) to touch that natal shore. "
            "Oh, snatch some portion of these acts from fate, "
            "Celestial Muse! and to our world relate."
        ),
    },
    {
        "id": "cowper_odyssey",
        "epic": "odyssey",
        "language": "english",
        "translator": "cowper",
        "verse_form": "blank_verse",
        "text": (
            "Muse make the man thy theme, for shrewdness famed "
            "And genius versatile, who far and wide "
            "A Wand'rer, after Ilium overthrown, "
            "Discover'd various cities, and the mind "
            "And manners learn'd of men, in lands remote. "
            "He num'rous woes on Ocean toss'd, endured, "
            "Anxious to save himself, and to conduct "
            "His followers to their home; yet all his care "
            "Preserved them not; they perish'd self-destroy'd "
            "By their own fault; infatuate! who devoured "
            "The oxen of the all-o'erseeing Sun, "
            "And, punish'd for that crime, return'd no more. "
            "Daughter divine of Jove, these things record, "
            "As it may please thee, even in our ears."
        ),
    },
    {
        "id": "butler_odyssey",
        "epic": "odyssey",
        "language": "english",
        "translator": "butler",
        "verse_form": "prose",
        "text": (
            "Tell me, O Muse, of that ingenious hero who travelled far and wide "
            "after he had sacked the famous town of Troy. Many cities did he "
            "visit, and many were the nations with whose manners and customs he "
            "was acquainted; moreover he suffered much by sea while trying to save "
            "his own life and bring his men safely home; but do what he might he "
            "could not save his men, for they perished through their own sheer "
            "folly in eating the cattle of the Sun-god Hyperion; so the god "
            "prevented them from ever reaching home."
        ),
    },
    {
        "id": "murray_odyssey",
        "epic": "odyssey",
        "language": "english",
        "translator": "murray",
        "verse_form": "prose",
        "text": (
            "Tell me, O Muse, of the man of many devices, who wandered full many "
            "ways after he had sacked the sacred citadel of Troy. Many were the "
            "men whose cities he saw and whose mind he learned, aye, and many the "
            "woes he suffered in his heart upon the sea, seeking to win his own "
            "life and the return of his comrades. Yet even so he saved not his "
            "comrades, though he desired it sore, for through their own blind "
            "folly they perished — fools, who devoured the kine of Helios "
            "Hyperion; but he took from them the day of their returning. Of these "
            "things, goddess, daughter of Zeus, beginning where thou wilt, tell "
            "thou even unto us."
        ),
    },
]


# ─────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────

def load_model():
    """Load Llama 3.1 70B Instruct in 4-bit quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"[{datetime.now():%H:%M:%S}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"[{datetime.now():%H:%M:%S}] Loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    print(f"[{datetime.now():%H:%M:%S}] Model loaded. "
          f"Layers: {model.config.num_hidden_layers}, "
          f"Hidden dim: {model.config.hidden_size}")

    return model, tokenizer


# ─────────────────────────────────────────────────
# ACTIVATION EXTRACTION
# ─────────────────────────────────────────────────

def extract_activations(model, tokenizer, passages):
    """
    Extract mean-pooled residual stream activations at every layer.

    Returns:
        activations: dict mapping passage_id -> np.array of shape (n_layers+1, hidden_dim)
                     Index 0 = embedding layer, 1..n_layers = transformer layers
        token_counts: dict mapping passage_id -> int
    """
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    activations = {}
    token_counts = {}

    for i, p in enumerate(passages):
        pid = p["id"]
        text = p["text"]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_tokens = inputs["input_ids"].shape[1]
        token_counts[pid] = n_tokens

        print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(passages)}] "
              f"{pid}: {n_tokens} tokens")

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states is a tuple of (n_layers+1) tensors
        # Each tensor: (batch=1, seq_len, hidden_dim)
        # Index 0 = embedding output, 1..n_layers = each transformer layer output
        hidden_states = outputs.hidden_states

        # Mean-pool across token dimension, convert to float32 for numerical stability
        layer_means = np.zeros((len(hidden_states), hidden_dim), dtype=np.float32)
        for layer_idx, hs in enumerate(hidden_states):
            # hs shape: (1, seq_len, hidden_dim)
            layer_means[layer_idx] = hs[0].float().mean(dim=0).cpu().numpy()

        activations[pid] = layer_means

        # Free memory
        del outputs, hidden_states, inputs
        torch.cuda.empty_cache()

    return activations, token_counts


# ─────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────

def run_analysis(activations, passages, output_dir):
    """PCA visualization and LOO classification at each layer."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(output_dir, exist_ok=True)

    # Build metadata arrays aligned with passage order
    ids = [p["id"] for p in passages]
    epics = np.array([p["epic"] for p in passages])
    languages = np.array([p["language"] for p in passages])
    translators = np.array([p["translator"] for p in passages])
    verse_forms = np.array([p["verse_form"] for p in passages])

    # Content labels: 0=iliad, 1=odyssey
    content_labels = np.array([0 if p["epic"] == "iliad" else 1 for p in passages])

    n_passages = len(passages)
    n_layers_plus_embed = activations[ids[0]].shape[0]
    hidden_dim = activations[ids[0]].shape[1]

    print(f"\n=== ANALYSIS ===")
    print(f"Passages: {n_passages}, Layers (incl embed): {n_layers_plus_embed}, "
          f"Hidden dim: {hidden_dim}")

    # Stack into matrix: (n_passages, hidden_dim) per layer
    all_acts = np.stack([activations[pid] for pid in ids])
    # Shape: (n_passages, n_layers+1, hidden_dim)

    # ── LOO Classification at each layer ──
    loo_results = []
    loo = LeaveOneOut()

    for layer_idx in range(n_layers_plus_embed):
        X = all_acts[:, layer_idx, :]  # (n_passages, hidden_dim)

        # Normalize per layer for stable classification
        X_norm = X - X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X_norm = X_norm / std

        # 1-NN LOO on content (iliad vs odyssey)
        correct = 0
        for train_idx, test_idx in loo.split(X_norm):
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_norm[train_idx], content_labels[train_idx])
            pred = clf.predict(X_norm[test_idx])
            correct += (pred == content_labels[test_idx]).sum()

        acc = correct / n_passages
        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"
        loo_results.append({"layer": layer_name, "layer_idx": layer_idx, "accuracy": acc})

        if layer_idx % 10 == 0 or acc >= 0.85:
            print(f"  Layer {layer_name:>10s}: LOO accuracy = {acc:.3f} "
                  f"({correct}/{n_passages})")

    # ── Save LOO results ──
    loo_path = os.path.join(output_dir, "loo_content_accuracy.json")
    with open(loo_path, "w") as f:
        json.dump(loo_results, f, indent=2)
    print(f"\nLOO results saved to {loo_path}")

    # ── LOO accuracy curve plot ──
    fig, ax = plt.subplots(figsize=(14, 5))
    layers = [r["layer_idx"] for r in loo_results]
    accs = [r["accuracy"] for r in loo_results]
    ax.plot(layers, accs, "o-", markersize=3, linewidth=1.2, color="#2c3e50")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("LOO Content Classification Accuracy")
    ax.set_title("Iliad vs Odyssey: 1-NN LOO Accuracy by Layer (n=14)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loo_accuracy_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Accuracy curve saved.")

    # ── PCA plots at selected layers ──
    # Pick: embed, layer 1, then every 10th, plus the peak accuracy layer
    peak_layer = max(loo_results, key=lambda r: r["accuracy"])["layer_idx"]
    plot_layers = sorted(set([0, 1, 10, 20, 30, 40, 50, 60, 70, peak_layer,
                              n_layers_plus_embed - 1]))
    plot_layers = [l for l in plot_layers if l < n_layers_plus_embed]

    # Color/marker schemes
    epic_colors = {"iliad": "#c0392b", "odyssey": "#2980b9"}
    lang_markers = {"greek": "D", "english": "o"}  # diamond vs circle
    translator_colors = {
        "homer": "#2c3e50",
        "chapman": "#e74c3c",
        "pope": "#3498db",
        "cowper": "#27ae60",
        "butler": "#f39c12",
        "murray": "#9b59b6",
    }

    for layer_idx in plot_layers:
        X = all_acts[:, layer_idx, :]
        pca = PCA(n_components=min(3, n_passages))
        X_pca = pca.fit_transform(X)

        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"
        acc = loo_results[layer_idx]["accuracy"]

        # ── Plot 1: Color by epic, shape by language ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        ax.set_title(f"{layer_name} — Color=Epic, Shape=Language\n"
                     f"LOO acc={acc:.2f}  |  "
                     f"PC1={pca.explained_variance_ratio_[0]:.1%}  "
                     f"PC2={pca.explained_variance_ratio_[1]:.1%}")
        for i in range(n_passages):
            ax.scatter(
                X_pca[i, 0], X_pca[i, 1],
                c=epic_colors[epics[i]],
                marker=lang_markers[languages[i]],
                s=120, edgecolors="black", linewidths=0.5, zorder=3,
            )
            ax.annotate(
                ids[i].replace("_iliad", "_I").replace("_odyssey", "_O"),
                (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6, alpha=0.7, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(True, alpha=0.2)

        # Legend for plot 1
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
                   markersize=10, label="Iliad"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
                   markersize=10, label="Odyssey"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markersize=8, label="Greek"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=10, label="English"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=8)

        # ── Plot 2: Color by translator ──
        ax = axes[1]
        ax.set_title(f"{layer_name} — Color=Translator")
        for i in range(n_passages):
            ax.scatter(
                X_pca[i, 0], X_pca[i, 1],
                c=translator_colors[translators[i]],
                marker="D" if epics[i] == "iliad" else "o",
                s=120, edgecolors="black", linewidths=0.5, zorder=3,
            )
            ax.annotate(
                ids[i].replace("_iliad", "_I").replace("_odyssey", "_O"),
                (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6, alpha=0.7, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(True, alpha=0.2)

        # Legend for plot 2
        legend_elements_2 = [
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=translator_colors[t],
                   markersize=10, label=t.capitalize())
            for t in ["homer", "chapman", "pope", "cowper", "butler", "murray"]
        ]
        legend_elements_2 += [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markersize=8, label="Iliad"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=10, label="Odyssey"),
        ]
        ax.legend(handles=legend_elements_2, loc="best", fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pca_{layer_name}.png"), dpi=150)
        plt.close(fig)

    print(f"PCA plots saved for {len(plot_layers)} layers.")

    # ── Cosine similarity matrices at key layers ──
    from scipy.spatial.distance import cosine

    for layer_idx in [0, peak_layer, n_layers_plus_embed - 1]:
        X = all_acts[:, layer_idx, :]
        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"

        sim_matrix = np.zeros((n_passages, n_passages))
        for i in range(n_passages):
            for j in range(n_passages):
                sim_matrix[i, j] = 1 - cosine(X[i], X[j])

        fig, ax = plt.subplots(figsize=(10, 8))
        import seaborn as sns
        short_ids = [pid.replace("_iliad", "_I").replace("_odyssey", "_O")
                     for pid in ids]
        sns.heatmap(
            sim_matrix, xticklabels=short_ids, yticklabels=short_ids,
            annot=True, fmt=".2f", cmap="RdBu_r", center=0.5,
            ax=ax, square=True, annot_kws={"size": 6},
        )
        ax.set_title(f"Cosine Similarity — {layer_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cosine_sim_{layer_name}.png"), dpi=150)
        plt.close(fig)

    print(f"Cosine similarity matrices saved.")

    # ── Summary statistics ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_passages": n_passages,
        "n_layers": n_layers_plus_embed - 1,
        "hidden_dim": hidden_dim,
        "peak_loo_layer": peak_layer,
        "peak_loo_accuracy": max(r["accuracy"] for r in loo_results),
        "chance_level": 0.5,
        "loo_accuracies_by_layer": {r["layer"]: r["accuracy"] for r in loo_results},
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return summary


# ─────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────

def print_diagnostics(token_counts, passages):
    """Print token count diagnostics — flag if pooling might be unstable."""
    print("\n=== TOKEN COUNT DIAGNOSTICS ===")
    counts = [(p["id"], token_counts[p["id"]]) for p in passages]
    counts.sort(key=lambda x: x[1])

    for pid, n in counts:
        lang = [p for p in passages if p["id"] == pid][0]["language"]
        flag = " ⚠️  SHORT" if n < 20 else ("  ⚠️  LONG" if n > 300 else "")
        print(f"  {pid:>20s}: {n:4d} tokens  [{lang}]{flag}")

    all_counts = [c[1] for c in counts]
    ratio = max(all_counts) / min(all_counts)
    print(f"\n  Range: {min(all_counts)} – {max(all_counts)} "
          f"(ratio {ratio:.1f}x)")

    if ratio > 5:
        print("  ⚠️  FLAG: Token count ratio >5x. Mean-pooling may be "
              "unstable — consider passage length adjustment after run one.")
    else:
        print("  ✓ Token count ratio acceptable for mean-pooling.")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HOMER PROEM RESIDUAL STREAM EXPERIMENT")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    output_dir = "homer_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save corpus for reproducibility
    corpus_path = os.path.join(output_dir, "corpus.json")
    with open(corpus_path, "w") as f:
        json.dump(PASSAGES, f, indent=2, ensure_ascii=False)
    print(f"Corpus saved to {corpus_path}")

    # Load model
    model, tokenizer = load_model()

    # Extract activations
    print(f"\n=== EXTRACTING ACTIVATIONS ===")
    activations, token_counts = extract_activations(model, tokenizer, PASSAGES)

    # Diagnostics
    print_diagnostics(token_counts, PASSAGES)

    # Save raw activations
    act_path = os.path.join(output_dir, "activations.npz")
    np.savez_compressed(
        act_path,
        **{pid: act for pid, act in activations.items()},
    )
    print(f"\nRaw activations saved to {act_path}")

    # Free model memory before analysis
    del model
    torch.cuda.empty_cache()
    print("Model freed from GPU memory.")

    # Run analysis
    summary = run_analysis(activations, PASSAGES, output_dir)

    # Print headline results
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    print(f"Peak LOO accuracy: {summary['peak_loo_accuracy']:.3f} "
          f"at layer {summary['peak_loo_layer']}")
    print(f"Chance level: {summary['chance_level']}")

    # Quick check: which passages were misclassified at peak?
    peak_idx = summary["peak_loo_layer"]
    ids = [p["id"] for p in PASSAGES]
    content_labels = np.array([0 if p["epic"] == "iliad" else 1 for p in PASSAGES])
    all_acts = np.stack([activations[pid] for pid in ids])
    X = all_acts[:, peak_idx, :]
    X_norm = X - X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_norm = X_norm / std

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut

    misclassified = []
    for train_idx, test_idx in LeaveOneOut().split(X_norm):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_norm[train_idx], content_labels[train_idx])
        pred = clf.predict(X_norm[test_idx])
        if pred != content_labels[test_idx]:
            misclassified.append(ids[test_idx[0]])

    if misclassified:
        print(f"\nMisclassified at peak layer: {misclassified}")
    else:
        print(f"\nAll 14 passages correctly classified at peak layer.")

    print(f"\nAll outputs in: {os.path.abspath(output_dir)}/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
