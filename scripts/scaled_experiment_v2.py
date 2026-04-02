"""
Scaled Experiment: 10 AK + 10 Notes (hand-curated)
====================================================
"""

import torch
import numpy as np
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from scipy.stats import binom

OUTPUT_DIR = "scaled_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AK_PASSAGES = [
    "Happy families are all alike; every unhappy family is unhappy in its own way. Everything was in confusion in the Oblonsky household. The wife had discovered that the husband was carrying on an intrigue with a French girl, who had been a governess in their family, and she had announced to her husband that she could not go on living in the same house with him. This position of affairs had now lasted three days, and not only the husband and wife themselves, but all the members of their family and household, were painfully conscious of it.",

    "And so Liberalism had become a habit of Stepan Arkadyevitch's, and he liked his newspaper, as he did his cigar after dinner, for the slight fog it diffused in his brain. He read the leading article, in which it was maintained that it was quite senseless in our day to raise an outcry that radicalism was threatening to swallow up all conservative elements.",

    "The petitioner, the widow of a staff captain Kalinin, came with a request impossible and unreasonable; but Stepan Arkadyevitch, as he generally did, made her sit down, heard her to the end attentively without interrupting her, and gave her detailed advice as to how and to whom to apply.",

    "It's the young people have to marry; and not their parents; and so we ought to leave the young people to arrange it as they choose. It was very easy for anyone to say that who had no daughters, but the princess realized that in the process of getting to know each other, her daughter might fall in love, and fall in love with someone who did not care to marry her or who was quite unfit to be her husband.",

    "The engine had already whistled in the distance. A few instants later the platform was quivering, and with puffs of steam hanging low in the air from the frost, the engine rolled up, with the lever of the middle wheel rhythmically moving up and down, and the stooping figure of the engine-driver covered with frost.",

    "He begged pardon, and was getting into the carriage, but felt he must glance at her once more; not that she was very beautiful, not on account of the elegance and modest grace which were apparent in her whole figure, but because in the expression of her charming face, as she passed close by him, there was something peculiarly caressing and soft.",

    "While they were still engaged, he had been struck by the definiteness with which she had declined the tour abroad and decided to go into the country, as though she knew of something she wanted, and could still think of something outside her love. This had jarred upon him then, and now her trivial cares and anxieties jarred upon him several times.",

    "Carefully set to rights, with hair well-brushed, in a smart little cap with some blue in it, her arms out on the quilt, she was lying on her back. Meeting his eyes, her eyes drew him to her. Her face, bright before, brightened still more as he drew near her. There was the same change in it from earthly to unearthly that is seen in the face of the dead.",

    "Alexey Alexandrovitch questioned him as to the duties of this new committee, and pondered. He was considering whether the new committee would not be acting in some way contrary to the views he had been advocating.",

    "His children? In Petersburg children did not prevent their parents from enjoying life. The children were brought up in schools, and there was no trace of the wild idea that prevailed in Moscow, in Lvov's household, for instance, that all the luxuries of life were for the children, while the parents have nothing but work and anxiety. Here people understood that a man is in duty bound to live for himself, as every man of culture should live.",
]

NOTES_PASSAGES = [
    "I am a sick man. I am a spiteful man. I am an unattractive man. I believe my liver is diseased. However, I know nothing at all about my disease, and do not know for certain what ails me. I don't consult a doctor for it, and never have, though I have a respect for medicine and doctors. Besides, I am extremely superstitious, sufficiently so to respect medicine, at any rate.",

    "When petitioners used to come for information to the table at which I sat, I used to grind my teeth at them, and felt intense enjoyment when I succeeded in making anybody unhappy. I almost did succeed. For the most part they were all timid people — of course, they were petitioners. But of the uppish ones there was one officer in particular I could not endure.",

    "That is really it. Observe yourselves more carefully, gentlemen, then you will understand that it is so. I invented adventures for myself and made up a life, so as at least to live in some way. How many times it has happened to me — well, for instance, to take offence simply on purpose, for nothing.",

    "Take North America — the eternal union. Take the farce of Schleswig-Holstein. And what is it that civilisation softens in us? The only gain of civilisation for mankind is the greater capacity for variety of sensations — and absolutely nothing more.",

    "They say that Cleopatra was fond of sticking gold pins into her slave-girls' breasts and derived gratification from their screams and writhings. You will say that that was in the comparatively barbarous times; that these are barbarous times too, because also, comparatively speaking, pins are stuck in even now.",

    "And if he does not find means he will contrive destruction and chaos, will contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse, may be by his curse alone he will attain his object — that is, convince himself that he is a man and not a piano-key!",

    "Kostanzhoglos and Uncle Pyotr Ivanitchs and foolishly accepting them as our ideal; they have slandered our romantics, taking them for the same transcendental sort as in Germany or France. On the contrary, the characteristics of our romantics are absolutely and directly opposed to the transcendental European type.",

    "The point was that I had attained my object, I had kept up my dignity, I had not yielded a step, and had put myself publicly on an equal social footing with him. I returned home feeling that I was fully avenged for everything. I was delighted. I was triumphant and sang Italian arias.",

    "Of course, I hated my fellow clerks one and all, and I despised them all, yet at the same time I was, as it were, afraid of them. In fact, it happened at times that I thought more highly of them than of myself. It somehow happened quite suddenly that I alternated between despising them and thinking them superior to myself.",

    "We Russians, speaking generally, have never had those foolish transcendental romantics — German, and still more French — on whom nothing produces any effect; if there were an earthquake, if all France perished at the barricades, they would still be the same, they would not even have the decency to affect a change.",
]

RUSSIAN_PROBES = [
    ("ak_ru_1", "ak", "Все счастливые семьи похожи друг на друга, каждая несчастливая "
     "семья несчастлива по-своему. Всё смешалось в доме Облонских. Жена "
     "узнала, что муж был в связи с бывшею в их доме француженкою-гувернанткой, "
     "и объявила мужу, что не может жить с ним в одном доме."),
    ("ak_ru_2", "ak", "В конце зимы, у Щербацких, происходил консилиум, долженствовавший "
     "решить, в каком положении находится здоровье Кити и что нужно "
     "предпринять для восстановления ее ослабевающих сил. Она была "
     "больна, и с приближением весны здоровье ее становилось хуже."),
    ("ak_ru_3", "ak", "Левин был женат третий месяц. Он был счастлив, но совсем не так, "
     "как ожидал. На каждом шагу он находил разочарование в прежних "
     "мечтах и новое, неожиданное очарование."),
    ("notes_ru_1", "notes", "Я человек больной... Я злой человек. Непривлекательный я "
     "человек. Я думаю, что у меня болит печень. Впрочем, я ни шиша "
     "не смыслю в моей болезни и не знаю наверно, что у меня болит."),
    ("notes_ru_2", "notes", "Я не только злым, но даже и ничем не сумел сделаться: ни злым, "
     "ни добрым, ни подлецом, ни честным, ни героем, ни насекомым. "
     "Теперь же доживаю в своем углу, дразня себя злобной и ни к чему "
     "не служащей утехой."),
    ("notes_ru_3", "notes", "Тогда мне было всего двадцать четыре года. Жизнь моя была уже "
     "и тогда угрюмая, беспорядочная и до одичалости одинокая. "
     "Я ни с кем не водился и даже избегал говорить и всё более и "
     "более забивался в свой угол."),
]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("=" * 70)
    print("SCALED EXPERIMENT: 10 AK + 10 Notes + 6 Russian probes")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    for model_id, quant in [
        ("meta-llama/Llama-3.1-70B-Instruct", True),
        ("meta-llama/Llama-3.1-70B", True),
        ("mistralai/Mistral-7B-Instruct-v0.3", False),
    ]:
        try:
            print(f"\n[{datetime.now():%H:%M:%S}] Trying {model_id}...")
            if quant:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                              device_map="auto", torch_dtype=torch.float16)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.eval()
            loaded_model = model_id
            print(f"[{datetime.now():%H:%M:%S}] Loaded {model_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    hidden = model.config.hidden_size

    def get_acts(text):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        ntok = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        layers = np.zeros((len(out.hidden_states), hidden), dtype=np.float32)
        for li, hs in enumerate(out.hidden_states):
            layers[li] = hs[0].float().mean(dim=0).cpu().numpy()
        del out, inputs; torch.cuda.empty_cache()
        return layers, ntok

    # Extract
    print("\n--- AK English ---")
    ak_acts = []
    for i, p in enumerate(AK_PASSAGES):
        acts, ntok = get_acts(p)
        ak_acts.append(acts)
        print(f"  AK_{i+1}: {ntok} tok | {p[:60]}...")

    print("\n--- Notes English ---")
    notes_acts = []
    for i, p in enumerate(NOTES_PASSAGES):
        acts, ntok = get_acts(p)
        notes_acts.append(acts)
        print(f"  Notes_{i+1}: {ntok} tok | {p[:60]}...")

    print("\n--- Russian probes ---")
    ru_data = []
    for pid, work, text in RUSSIAN_PROBES:
        acts, ntok = get_acts(text)
        ru_data.append((pid, work, acts))
        print(f"  {pid}: {ntok} tok")

    del model; torch.cuda.empty_cache()

    n_layers = ak_acts[0].shape[0]
    n_ak = len(ak_acts)
    n_notes = len(notes_acts)

    # ═══════════════════════════════════════
    # TEST 1: Within-work vs cross-work similarity
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TEST 1: ENGLISH CLUSTERING ({n_ak} AK, {n_notes} Notes)")
    print("="*70)
    print(f"{'Layer':>8s}  {'Within-AK':>10s}  {'Within-N':>10s}  {'Cross':>10s}  {'Gap':>8s}")
    print("-" * 52)

    for li in range(0, n_layers, max(1, n_layers//16)):
        w_ak = [1-cosine(ak_acts[i][li], ak_acts[j][li])
                for i in range(n_ak) for j in range(i+1, n_ak)]
        w_n = [1-cosine(notes_acts[i][li], notes_acts[j][li])
               for i in range(n_notes) for j in range(i+1, n_notes)]
        cross = [1-cosine(ak_acts[i][li], notes_acts[j][li])
                 for i in range(n_ak) for j in range(n_notes)]
        within = np.mean(w_ak + w_n)
        cross_m = np.mean(cross)
        nm = "embed" if li == 0 else f"L{li}"
        print(f"  {nm:>8s}  {np.mean(w_ak):>10.4f}  {np.mean(w_n):>10.4f}  {cross_m:>10.4f}  {within-cross_m:>+8.4f}")

    # ═══════════════════════════════════════
    # TEST 2: Leave-one-out (English, n=20)
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 2: LEAVE-ONE-OUT CLASSIFICATION (n=20)")
    print("="*70)

    all_en = [(f"ak_{i}", "ak", ak_acts[i]) for i in range(n_ak)] + \
             [(f"notes_{i}", "notes", notes_acts[i]) for i in range(n_notes)]

    print(f"{'Layer':>8s}  {'Correct':>8s}  {'Acc':>6s}  {'p':>12s}")
    print("-" * 40)

    for li in range(0, n_layers, max(1, n_layers//16)):
        correct = 0
        for idx in range(len(all_en)):
            pid, work, act = all_en[idx]
            ak_c = np.mean([a[li] for p, w, a in all_en if w == "ak" and p != pid], axis=0)
            n_c = np.mean([a[li] for p, w, a in all_en if w == "notes" and p != pid], axis=0)
            pred = "ak" if (1-cosine(act[li], ak_c)) > (1-cosine(act[li], n_c)) else "notes"
            if pred == work: correct += 1
        p = binom.sf(correct-1, 20, 0.5)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        nm = "embed" if li == 0 else f"L{li}"
        print(f"  {nm:>8s}  {correct:>5d}/20  {correct/20:>5.0%}  {p:>12.8f} {sig}")

    # ═══════════════════════════════════════
    # TEST 3: Cross-lingual classification
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 3: CROSS-LINGUAL CLASSIFICATION (6 Russian probes)")
    print("="*70)

    print(f"{'Layer':>8s}  {'Correct':>8s}  {'Details':>40s}")
    print("-" * 55)

    for li in range(0, n_layers, max(1, n_layers//16)):
        ak_cent = np.mean([a[li] for a in ak_acts], axis=0)
        notes_cent = np.mean([a[li] for a in notes_acts], axis=0)
        correct = 0
        details = []
        for pid, work, acts in ru_data:
            sim_ak = 1-cosine(acts[li], ak_cent)
            sim_n = 1-cosine(acts[li], notes_cent)
            pred = "ak" if sim_ak > sim_n else "notes"
            ok = pred == work
            if ok: correct += 1
            details.append(f"{pid}={'✓' if ok else '✗'}")
        nm = "embed" if li == 0 else f"L{li}"
        print(f"  {nm:>8s}  {correct:>5d}/6   {' '.join(details)}")

    # ═══════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════
    np.savez_compressed(os.path.join(OUTPUT_DIR, "all_acts.npz"),
        **{f"ak_en_{i}": ak_acts[i] for i in range(n_ak)},
        **{f"notes_en_{i}": notes_acts[i] for i in range(n_notes)},
        **{pid: acts for pid, work, acts in ru_data},
    )

    print(f"\n{'='*70}")
    print(f"Model: {loaded_model}")
    print(f"AK passages: {n_ak}, Notes passages: {n_notes}, Russian probes: {len(ru_data)}")
    print(f"Done: {datetime.now()}")
    print("="*70)


if __name__ == "__main__":
    main()
