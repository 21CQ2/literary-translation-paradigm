"""
Multi-Passage + Base Model Replication
=======================================
Addresses the fresh reviewer's top concerns:
1. Multiple passages per work (M1)
2. Base model check (M4)
3. Clean bootstrap pool (strip same-author contamination)

Design: 3 passages from AK + 3 from Notes, each in Russian + English = 12 passages
Then re-run with base model on the original 4 Experiment 2 passages.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
from matplotlib.lines import Line2D

OUTPUT_DIR = "multipassage_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# MULTI-PASSAGE CORPUS
# 3 passages per work × 2 languages = 12
# ═══════════════════════════════════════════════

PASSAGES = [
    # ── AK Part 1 Ch 1 (opening) ──
    {"id": "ak1_ru", "work": "ak", "language": "russian",
     "text": "Все счастливые семьи похожи друг на друга, каждая несчастливая "
             "семья несчастлива по-своему. Всё смешалось в доме Облонских. "
             "Жена узнала, что муж был в связи с бывшею в их доме "
             "француженкою-гувернанткой, и объявила мужу, что не может жить "
             "с ним в одном доме. Положение это продолжалось уже третий день "
             "и мучительно чувствовалось и самими супругами, и всеми членами "
             "семьи, и домочадцами."},
    {"id": "ak1_en", "work": "ak", "language": "english",
     "text": "Happy families are all alike; every unhappy family is unhappy "
             "in its own way. Everything was in confusion in the Oblonsky "
             "household. The wife had discovered that the husband was carrying "
             "on an intrigue with a French girl, who had been a governess in "
             "their family, and she had announced to her husband that she "
             "could not go on living in the same house with him. This position "
             "of affairs had now lasted three days, and not only the husband "
             "and wife themselves, but all the members of their family and "
             "household, were painfully conscious of it."},

    # ── AK Part 2 Ch 1 ──
    {"id": "ak2_ru", "work": "ak", "language": "russian",
     "text": "В конце зимы, у Щербацких, происходил консилиум, долженствовавший "
             "решить, в каком положении находится здоровье Кити и что нужно "
             "предпринять для восстановления ее ослабевающих сил. Она была "
             "больна, и с приближением весны здоровье ее становилось хуже. "
             "Домашний доктор давал ей рыбий жир, потом железо, потом ляпис, "
             "но так как ни то, ни другое, ни третье не помогало и так как "
             "он советовал на весну поехать за границу, то приглашен был "
             "знаменитый доктор."},
    {"id": "ak2_en", "work": "ak", "language": "english",
     "text": "At the end of the winter, in the Shtcherbatskys' house, a "
             "consultation was being held, which was to pronounce on the state "
             "of Kitty's health and the measures to be taken to restore her "
             "failing strength. She had been ill, and as spring came on she "
             "grew worse. The family doctor gave her cod liver oil, then iron, "
             "then nitrate of silver, but as the first and the second and the "
             "third were alike in doing no good, and as his advice when spring "
             "came was to go abroad, a celebrated physician was called in."},

    # ── AK Part 5 Ch 1 ──
    {"id": "ak3_ru", "work": "ak", "language": "russian",
     "text": "Княгиня Щербацкая находила, что свадьбу до поста, до которого "
             "оставалось пять недель, нельзя было справить, так как половина "
             "приданого не могла поспеть к этому времени; но она не могла не "
             "согласиться с Левиным, что после поста было бы уже и слишком "
             "поздно, так как старая родственница князя Щербацкого была "
             "очень больна и могла скоро умереть, и тогда траур задержал бы "
             "еще свадьбу."},
    {"id": "ak3_en", "work": "ak", "language": "english",
     "text": "Princess Shtcherbatskaya considered that it was out of the "
             "question to have the wedding before Lent, just five weeks off, "
             "since not half the trousseau could possibly be ready by that "
             "time; but she could not but agree with Levin that to put it off "
             "till after Lent would be putting it off too late, as an old aunt "
             "of Prince Shtcherbatsky's was seriously ill and might die, and "
             "then the mourning would delay the wedding still longer."},

    # ── Notes Part 1 Section 1 (opening) ──
    {"id": "notes1_ru", "work": "notes", "language": "russian",
     "text": "Я человек больной... Я злой человек. Непривлекательный я "
             "человек. Я думаю, что у меня болит печень. Впрочем, я ни шиша "
             "не смыслю в моей болезни и не знаю наверно, что у меня болит. "
             "Я не лечусь и никогда не лечился, хотя медицину и докторов "
             "уважаю. К тому же я еще и суеверен до крайности; ну, хоть "
             "настолько, чтоб уважать медицину. Я достаточно образован, чтоб "
             "не быть суеверным, но я суеверен. Нет-с, я не хочу лечиться "
             "со злости. Вот этого вы, наверно, не изволите понимать. Ну-с, "
             "а я понимаю."},
    {"id": "notes1_en", "work": "notes", "language": "english",
     "text": "I am a sick man... I am a spiteful man. I am an unattractive "
             "man. I believe my liver is diseased. However, I know nothing at "
             "all about my disease, and do not know for certain what ails me. "
             "I don't consult a doctor for it, and never have, though I have "
             "a respect for medicine and doctors. Besides, I am extremely "
             "superstitious, sufficiently so to respect medicine, at any rate. "
             "I am well-educated enough not to be superstitious, but I am "
             "superstitious. No, I refuse to consult a doctor from spite. "
             "That you probably will not understand. Well, I understand it "
             "though."},

    # ── Notes Part 1 Section 2 ──
    {"id": "notes2_ru", "work": "notes", "language": "russian",
     "text": "Я не только злым, но даже и ничем не сумел сделаться: ни злым, "
             "ни добрым, ни подлецом, ни честным, ни героем, ни насекомым. "
             "Теперь доживаю в своем углу, дразня себя злобною и ни к чему "
             "не служащею утехой, что умный человек и не может серьезно "
             "чем-нибудь сделаться, а делается чем-нибудь только дурак."},
    {"id": "notes2_en", "work": "notes", "language": "english",
     "text": "I was not only unable to become spiteful, I did not know how to "
             "become anything; neither spiteful nor kind, neither a rascal nor "
             "an honest man, neither a hero nor an insect. Now, I am living "
             "out my life in my corner, taunting myself with the spiteful and "
             "useless consolation that an intelligent man cannot become "
             "anything seriously, and it is only the fool who becomes "
             "anything."},

    # ── Notes Part 2 Chapter 1 (opening) ──
    {"id": "notes3_ru", "work": "notes", "language": "russian",
     "text": "Тогда мне было всего двадцать четыре года. Жизнь моя была уже "
             "и тогда угрюмая, беспорядочная и до одичалости одинокая. "
             "Я ни с кем не водился и даже избегал говорить и всё более и "
             "более забивался в свой угол. На службе, в канцелярии, я "
             "старался даже не глядеть ни на кого, и я очень хорошо замечал, "
             "что на меня товарищи не только смотрели как на чудака, но "
             "и — мне всё казалось это — смотрели как будто с каким-то "
             "омерзением."},
    {"id": "notes3_en", "work": "notes", "language": "english",
     "text": "At that time I was only twenty-four. My life was even then "
             "gloomy, ill-regulated, and as solitary as that of a savage. "
             "I made friends with no one and positively avoided talking, and "
             "buried myself more and more in my hole. At work in the office "
             "I never looked at any one, and I was perfectly well aware that "
             "my companions looked upon me, not only as a queer fellow, but "
             "even — I always fancied this — with a sort of loathing."},
]


def run_multipassage(model_id, model_label, output_subdir):
    """Extract activations and run analysis for a given model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    outdir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(outdir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    print(f"\n[{datetime.now():%H:%M:%S}] Loading {model_label}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()
    hidden_dim = model.config.hidden_size

    # Extract
    activations = {}
    for i, p in enumerate(PASSAGES):
        pid = p["id"]
        inputs = tokenizer(p["text"], return_tensors="pt").to(model.device)
        n_tok = inputs["input_ids"].shape[1]
        print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(PASSAGES)}] {pid}: {n_tok} tokens")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        layer_means = np.zeros((len(outputs.hidden_states), hidden_dim), dtype=np.float32)
        for li, hs in enumerate(outputs.hidden_states):
            layer_means[li] = hs[0].float().mean(dim=0).cpu().numpy()
        activations[pid] = layer_means
        del outputs, inputs; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(outdir, "activations.npz"), **activations)

    # ── Analysis ──
    ids = [p["id"] for p in PASSAGES]
    works = [p["work"] for p in PASSAGES]
    langs = [p["language"] for p in PASSAGES]
    all_acts = np.stack([activations[pid] for pid in ids])
    n_layers = all_acts.shape[1]

    print(f"\n{'='*60}")
    print(f"MULTI-PASSAGE CONTENT TEST — {model_label}")
    print(f"{'='*60}")
    print("For each passage pair (Russian, English), does the Russian")
    print("cluster with same-work English rather than different-work English?\n")

    # At each layer, for each Russian passage, compute:
    #   sim to same-work English centroid vs different-work English centroid
    ak_ru_ids = ["ak1_ru", "ak2_ru", "ak3_ru"]
    ak_en_ids = ["ak1_en", "ak2_en", "ak3_en"]
    notes_ru_ids = ["notes1_ru", "notes2_ru", "notes3_ru"]
    notes_en_ids = ["notes1_en", "notes2_en", "notes3_en"]

    results = []
    for layer_idx in range(0, n_layers, 5):
        X = all_acts[:, layer_idx, :]
        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"

        # English centroids
        ak_en_centroid = np.stack([X[ids.index(i)] for i in ak_en_ids]).mean(0)
        notes_en_centroid = np.stack([X[ids.index(i)] for i in notes_en_ids]).mean(0)

        correct = 0
        total = 0
        margins = []
        for ru_id in ak_ru_ids:
            v = X[ids.index(ru_id)]
            sim_ak = 1 - cosine(v, ak_en_centroid)
            sim_notes = 1 - cosine(v, notes_en_centroid)
            if sim_ak > sim_notes: correct += 1
            margins.append(sim_ak - sim_notes)
            total += 1
        for ru_id in notes_ru_ids:
            v = X[ids.index(ru_id)]
            sim_notes = 1 - cosine(v, notes_en_centroid)
            sim_ak = 1 - cosine(v, ak_en_centroid)
            if sim_notes > sim_ak: correct += 1
            margins.append(sim_notes - sim_ak)
            total += 1

        mean_margin = float(np.mean(margins))
        min_margin = float(np.min(margins))
        acc = correct / total

        results.append({
            "layer": layer_name, "layer_idx": layer_idx,
            "accuracy": acc, "correct": correct, "total": total,
            "mean_margin": mean_margin, "min_margin": min_margin,
            "margins": [float(m) for m in margins],
        })

        if layer_idx % 10 == 0:
            check = "✓" if acc == 1.0 else f"{correct}/{total}"
            print(f"  {layer_name:>10s}: {check}  mean_Δ={mean_margin:+.3f}  "
                  f"min_Δ={min_margin:+.3f}")

    with open(os.path.join(outdir, "multipassage_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # PCA at key layers
    work_colors = {"ak": "#c0392b", "notes": "#2980b9"}
    lang_markers = {"russian": "D", "english": "o"}

    for layer_idx in [0, 20, 40, 60, 80]:
        if layer_idx >= n_layers: continue
        X = all_acts[:, layer_idx, :]
        pca = PCA(n_components=2).fit_transform(X)
        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"{layer_name} — Multi-Passage ({model_label})\n"
                     f"AK ●/◆ (red) vs Notes ●/◆ (blue)")
        for i in range(len(ids)):
            color = work_colors[works[i]]
            marker = lang_markers[langs[i]]
            ax.scatter(pca[i, 0], pca[i, 1], c=color, marker=marker,
                       s=180, edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(ids[i], (pca[i, 0], pca[i, 1]),
                        fontsize=6, alpha=0.8, ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
                   markersize=10, label="AK English"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b",
                   markersize=8, label="AK Russian"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
                   markersize=10, label="Notes English"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#2980b9",
                   markersize=8, label="Notes Russian"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pca_{layer_name}.png"), dpi=150)
        plt.close(fig)

    # Summary
    perfect_layers = sum(1 for r in results if r["accuracy"] == 1.0)
    print(f"\n  SUMMARY ({model_label}):")
    print(f"    Perfect classification (6/6): {perfect_layers}/{len(results)} layers")
    print(f"    Peak mean margin: {max(r['mean_margin'] for r in results):+.3f}")

    return results


def main():
    print("=" * 60)
    print("MULTI-PASSAGE + BASE MODEL REPLICATION")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    # 1. Multi-passage with Instruct model
    instruct_results = run_multipassage(
        "meta-llama/Llama-3.1-70B-Instruct",
        "Llama-3.1-70B-Instruct",
        "instruct",
    )

    # 2. Same analysis with Base model
    base_results = run_multipassage(
        "meta-llama/Llama-3.1-70B",
        "Llama-3.1-70B-Base",
        "base",
    )

    # 3. Comparison
    print("\n" + "=" * 60)
    print("INSTRUCT vs BASE COMPARISON")
    print("=" * 60)
    for layer_idx in range(0, 81, 10):
        ir = next((r for r in instruct_results if r["layer_idx"] == layer_idx), None)
        br = next((r for r in base_results if r["layer_idx"] == layer_idx), None)
        if ir and br:
            layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"
            print(f"  {layer_name:>10s}: Instruct {ir['correct']}/{ir['total']} "
                  f"Δ={ir['mean_margin']:+.3f}  |  Base {br['correct']}/{br['total']} "
                  f"Δ={br['mean_margin']:+.3f}")

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
