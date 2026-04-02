"""
Clean Null Pool: Diverse Unrelated Passages
============================================
The previous structured bootstrap was contaminated by Homer's
content structure (21/29 pool passages were Homer translations
with known Iliad/Odyssey clustering).

Fix: extract activations for 25 diverse, unrelated public domain
passages. No two share content, genre, or topic. Use ONLY these
as the null pool. Then re-run bootstrap for Experiments 2 and 3.

This gives a null that asks: "among passages with real residual
stream structure but no content relationship, how often do 4
random passages produce margins as large as observed?"
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

OUTPUT_DIR = "diverse_bootstrap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 25 diverse public domain opening paragraphs
# Mix of: fiction, nonfiction, philosophy, science, history
# Mix of: English, Russian, French, German originals
# No two share topic, genre, or content relationship

DIVERSE_PASSAGES = [
    {
        "id": "moby_dick",
        "text": "Call me Ishmael. Some years ago — never mind how long precisely — having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    },
    {
        "id": "pride_prejudice",
        "text": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
    },
    {
        "id": "tale_two_cities",
        "text": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
    },
    {
        "id": "crime_punishment_ru",
        "text": "В начале июля, в чрезвычайно жаркое время, под вечер, один молодой человек вышел из своей каморки, которую нанимал от жильцов в С-м переулке, на улицу и медленно, как бы в нерешимости, отправился к К-ну мосту.",
    },
    {
        "id": "crime_punishment_en",
        "text": "On an exceptionally hot evening early in July, a young man came out of the garret in which he lodged in S. Place and walked slowly, as though in hesitation, towards K. bridge.",
    },
    {
        "id": "communist_manifesto",
        "text": "A spectre is haunting Europe — the spectre of communism. All the powers of old Europe have entered into a holy alliance to exorcise this spectre: Pope and Tsar, Metternich and Guizot, French Radicals and German police-spies.",
    },
    {
        "id": "wealth_of_nations",
        "text": "The annual labour of every nation is the fund which originally supplies it with all the necessaries and conveniences of life which it annually consumes, and which consist always either in the immediate produce of that labour, or in what is purchased with that produce from other nations.",
    },
    {
        "id": "alice_wonderland",
        "text": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'",
    },
    {
        "id": "metamorphosis_de",
        "text": "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren Ungeziefer verwandelt. Er lag auf seinem panzerartig harten Rücken und sah, wenn er den Kopf ein wenig hob, seinen gewölbten, braunen, von bogenförmigen Versteifungen geteilten Bauch, auf dessen Höhe sich die Bettdecke, zum gänzlichen Niedergleiten bereit, kaum noch erhalten konnte.",
    },
    {
        "id": "metamorphosis_en",
        "text": "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections.",
    },
    {
        "id": "republic_plato",
        "text": "I went down yesterday to the Piraeus with Glaucon the son of Ariston, that I might offer up my prayers to the goddess; and also because I wanted to see in what manner they would celebrate the festival, which was a new thing. I was delighted with the procession of the inhabitants; but that of the Thracians was equally, if not more, beautiful.",
    },
    {
        "id": "principia_newton",
        "text": "Every body continues in its state of rest, or of uniform motion in a right line, unless it is compelled to change that state by forces impressed upon it. The change of motion is proportional to the motive force impressed; and is made in the direction of the right line in which that force is impressed.",
    },
    {
        "id": "walden_thoreau",
        "text": "When I wrote the following pages, or rather the bulk of them, I lived alone, in the woods, a mile from any neighbor, in a house which I had built myself, on the shore of Walden Pond, in Concord, Massachusetts, and earned my living by the labor of my hands only.",
    },
    {
        "id": "brothers_karamazov_ru",
        "text": "Алексей Федорович Карамазов был третий сын помещика нашего уезда Федора Павловича Карамазова, столь известного в свое время по одному темному и трагическому делу, которое будет поведано мною в надлежащем месте.",
    },
    {
        "id": "les_miserables_fr",
        "text": "En 1815, M. Charles-François-Bienvenu Myriel était évêque de Digne. C'était un vieillard d'environ soixante-quinze ans; il occupait le siège de Digne depuis 1806. Quoique ce détail ne touche en aucune manière au fond même de ce que nous avons à raconter, il n'est peut-être pas inutile, ne fût-ce que dans un intérêt d'exactitude en toutes choses.",
    },
    {
        "id": "candide_fr",
        "text": "Il y avait en Westphalie, dans le château de monsieur le baron de Thunder-ten-tronckh, un jeune garçon à qui la nature avait donné les mœurs les plus douces. Sa physionomie annonçait son âme. Il avait le jugement assez droit, avec l'esprit le plus simple; c'est, je crois, pour cette raison qu'on le nommait Candide.",
    },
    {
        "id": "faust_de",
        "text": "Habe nun, ach! Philosophie, Juristerei und Medizin, und leider auch Theologie durchaus studiert, mit heißem Bemühn. Da steh ich nun, ich armer Tor, und bin so klug als wie zuvor! Heiße Magister, heiße Doktor gar, und ziehe schon an die zehen Jahr herauf, herab und quer und krumm meine Schüler an der Nase herum.",
    },
    {
        "id": "gettysburg",
        "text": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.",
    },
    {
        "id": "don_quixote",
        "text": "In a village of La Mancha, the name of which I have no desire to call to mind, there lived not long since one of those gentlemen that keep a lance in the lance-rack, an old buckler, a lean hack, and a greyhound for coursing.",
    },
    {
        "id": "war_peace_ru",
        "text": "Ну, что, князь, Генуа и Лукка стали не больше, как поместьями фамилии Бонапарте. Нет, я вас предупреждаю, если вы мне не скажете, что у нас война, если вы ещё позволите себе защищать все гадости, все ужасы этого Антихриста, я вас больше не знаю, вы уже не друг мой.",
    },
    {
        "id": "meditations_marcus",
        "text": "From my grandfather Verus I learned good morals and the government of my temper. From the reputation and remembrance of my father, modesty and a manly character. From my mother, piety and beneficence, and abstinence, not only from evil deeds, but even from evil thoughts.",
    },
    {
        "id": "frankenstein",
        "text": "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings. I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking.",
    },
    {
        "id": "jungle_book",
        "text": "It was seven o'clock of a very warm evening in the Seeonee hills when Father Wolf woke up from his day's rest, scratched himself, yawned, and spread out his paws one after the other to get rid of the sleepy feeling in their tips.",
    },
    {
        "id": "eugene_onegin_ru",
        "text": "Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог. Его пример другим наука; но, боже мой, какая скука с больным сидеть и день и ночь, не отходя ни шагу прочь!",
    },
    {
        "id": "sherlock_holmes",
        "text": "Mr. Sherlock Holmes, who was usually very late in the mornings, save upon those not infrequent occasions when he was up all night, was seated at the breakfast table. I stood upon the hearth-rug and picked up the stick which our visitor had left behind him the night before.",
    },
]


def extract_diverse(model, tokenizer):
    hidden_dim = model.config.hidden_size
    activations = {}
    for i, p in enumerate(DIVERSE_PASSAGES):
        pid = p["id"]
        inputs = tokenizer(p["text"], return_tensors="pt").to(model.device)
        n_tokens = inputs["input_ids"].shape[1]
        print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(DIVERSE_PASSAGES)}] {pid}: {n_tokens} tokens")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        layer_means = np.zeros((len(outputs.hidden_states), hidden_dim), dtype=np.float32)
        for li, hs in enumerate(outputs.hidden_states):
            layer_means[li] = hs[0].float().mean(dim=0).cpu().numpy()
        activations[pid] = layer_means
        del outputs, inputs; torch.cuda.empty_cache()
    return activations


def run_clean_bootstrap(diverse_acts, exp_acts, experiment_name,
                         id_a1, id_a2, id_b1, id_b2, n_boot=10000):
    """Bootstrap from diverse unrelated passages only."""
    print(f"\n{'='*60}")
    print(f"CLEAN BOOTSTRAP: {experiment_name}")
    print(f"{'='*60}")

    pool_keys = list(diverse_acts.keys())
    n_pool = len(pool_keys)
    n_layers = diverse_acts[pool_keys[0]].shape[0]
    rng = np.random.RandomState(42)

    results = []
    for layer_idx in range(0, n_layers, 5):
        pool_vecs = np.stack([diverse_acts[k][layer_idx] for k in pool_keys])

        # Observed margin from experimental activations
        a1 = exp_acts[id_a1][layer_idx]
        a2 = exp_acts[id_a2][layer_idx]
        b1 = exp_acts[id_b1][layer_idx]
        b2 = exp_acts[id_b2][layer_idx]

        obs_margin = float(
            ((1 - cosine(a1, a2)) - (1 - cosine(a1, b2))) +
            ((1 - cosine(b1, b2)) - (1 - cosine(b1, a2)))
        )

        # Null: draw 4 from diverse pool, best of 3 pairings
        null_margins = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n_pool, size=4, replace=False)
            v1, v2, v3, v4 = pool_vecs[idx[0]], pool_vecs[idx[1]], pool_vecs[idx[2]], pool_vecs[idx[3]]

            def pair_margin(a1, a2, b1, b2):
                return ((1 - cosine(a1, a2)) - (1 - cosine(a1, b2))) + \
                       ((1 - cosine(b1, b2)) - (1 - cosine(b1, a2)))

            m1 = pair_margin(v1, v2, v3, v4)
            m2 = pair_margin(v1, v3, v2, v4)
            m3 = pair_margin(v1, v4, v2, v3)
            null_margins[b] = max(m1, m2, m3)

        p_value = float((np.sum(null_margins >= obs_margin) + 1) / (n_boot + 1))
        layer_name = "embed" if layer_idx == 0 else f"layer_{layer_idx}"

        results.append({
            "layer": layer_name, "layer_idx": layer_idx,
            "obs_margin": obs_margin,
            "null_mean": float(null_margins.mean()),
            "null_std": float(null_margins.std()),
            "null_95": float(np.percentile(null_margins, 95)),
            "null_max": float(null_margins.max()),
            "p_value": p_value,
        })

        if layer_idx % 10 == 0:
            sig = "***" if p_value < .001 else ("**" if p_value < .01 else ("*" if p_value < .05 else "ns"))
            print(f"  {layer_name:>10s}: obs={obs_margin:+.4f}  "
                  f"null={null_margins.mean():+.4f}±{null_margins.std():.4f}  "
                  f"95th={np.percentile(null_margins, 95):+.4f}  "
                  f"p={p_value:.4f} {sig}")

    # Save
    with open(os.path.join(OUTPUT_DIR, f"bootstrap_{experiment_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    layers = [r["layer_idx"] for r in results]
    obs_vals = [r["obs_margin"] for r in results]
    null_95 = [r["null_95"] for r in results]
    null_means = [r["null_mean"] for r in results]
    pvals = [r["p_value"] for r in results]

    ax1.plot(layers, obs_vals, "o-", color="#c0392b", markersize=4, linewidth=1.5, label="Observed")
    ax1.fill_between(layers,
                      [r["null_mean"] - 2*r["null_std"] for r in results],
                      [r["null_mean"] + 2*r["null_std"] for r in results],
                      alpha=0.2, color="#7f8c8d", label="Null ±2σ")
    ax1.plot(layers, null_95, "--", color="#e67e22", linewidth=1, label="Null 95th pct")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Sum of margins")
    ax1.set_title(f"Clean Bootstrap: {experiment_name}\n(null: {n_pool} diverse unrelated passages, {n_boot:,} samples)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=.3)

    ax2.plot(layers, [-np.log10(max(p, 1e-5)) for p in pvals],
             "o-", color="#2c3e50", markersize=4, linewidth=1.5)
    ax2.axhline(-np.log10(.05), color="red", ls="--", alpha=.5, label="p=0.05")
    ax2.axhline(-np.log10(.01), color="orange", ls="--", alpha=.5, label="p=0.01")
    ax2.axhline(-np.log10(.001), color="green", ls="--", alpha=.5, label="p=0.001")
    ax2.set_xlabel("Layer"); ax2.set_ylabel("-log10(p)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"bootstrap_{experiment_name}.png"), dpi=150)
    plt.close(fig)

    return results


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("=" * 60)
    print("CLEAN BOOTSTRAP WITH DIVERSE NULL POOL")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    # Load model
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    print(f"[{datetime.now():%H:%M:%S}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()

    # Extract diverse passages
    print(f"\n=== EXTRACTING {len(DIVERSE_PASSAGES)} DIVERSE PASSAGES ===")
    diverse_acts = extract_diverse(model, tokenizer)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "diverse_activations.npz"), **diverse_acts)

    del model; torch.cuda.empty_cache()
    print("Model freed.")

    # Load experimental activations
    td_data = np.load("propnoun_results/activations.npz")
    td_acts = {k: td_data[k] for k in td_data.files}

    gogol_data = np.load("gogol_results/activations.npz")
    gogol_acts = {k: gogol_data[k] for k in gogol_data.files}

    # Run bootstraps
    td_results = run_clean_bootstrap(
        diverse_acts, td_acts, "tolstoy_dostoevsky",
        "ak_russian", "ak_garnett", "notes_russian", "notes_garnett",
    )

    gogol_results = run_clean_bootstrap(
        diverse_acts, gogol_acts, "gogol",
        "ds_russian", "ds_garnett", "oc_russian", "oc_garnett",
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — CLEAN BOOTSTRAP")
    print("=" * 60)

    for name, results in [("Tolstoy-Dostoevsky", td_results), ("Gogol", gogol_results)]:
        sig = sum(1 for r in results if r["p_value"] < 0.05)
        best_p = min(r["p_value"] for r in results)
        peak_layer = max(results, key=lambda r: r["obs_margin"])
        print(f"\n  {name}:")
        print(f"    Significant (p<0.05): {sig}/{len(results)} layers")
        print(f"    Best p-value: {best_p:.4f}")
        print(f"    Peak: {peak_layer['layer']} obs={peak_layer['obs_margin']:+.4f} "
              f"null_95={peak_layer['null_95']:+.4f} p={peak_layer['p_value']:.4f}")

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
