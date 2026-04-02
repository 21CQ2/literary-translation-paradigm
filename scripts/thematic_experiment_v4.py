"""
Thematic Experiment v4 — Definitive Design
============================================
4 works in ISOLATION theme, 3 works in FAMILY theme, Kinder egg control.
3 languages. AK and CP cross both themes. BK crosses both themes.

ISOLATION (4 works × 3 langs = 12):
  CP: Raskolnikov in garret
  Notes: Underground Man shunning colleagues
  AK: Anna's despair on the train
  BK: Ivan confronting himself / the devil

FAMILY (3 works × 3 langs = 9):
  AK: Oblonsky household crisis
  BK: Fyodor Pavlovitch introduced
  CP: Marmeladov's family ruin

CONTROL (1 text × 3 langs = 3):
  Kinder Surprise choking hazard warning

Total: 24 passages.

Works crossing both themes: AK (isolation + family), BK (isolation + family), CP (isolation + family).
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

OUTPUT_DIR = "thematic_v4_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PASSAGES = [
    # ═══════════════════════════════════════
    # ISOLATION — CP (Raskolnikov)
    # ═══════════════════════════════════════
    {"id": "cp_isol_ru", "work": "cp", "theme": "isolation", "lang": "ru",
     "text": "В начале июля, в чрезвычайно жаркое время, под вечер, один "
             "молодой человек вышел из своей каморки, которую нанимал от "
             "жильцов в С-м переулке, на улицу и медленно, как бы в "
             "нерешимости, отправился к К-ну мосту. Он благополучно избежал "
             "встречи с своей хозяйкой на лестнице. Каморка его приходилась "
             "под самой кровлей высокого пятиэтажного дома и походила более "
             "на шкаф, чем на квартиру."},
    {"id": "cp_isol_en", "work": "cp", "theme": "isolation", "lang": "en",
     "text": "On an exceptionally hot evening early in July, a young man came "
             "out of the garret in which he lodged in S. Place and walked "
             "slowly, as though in hesitation, towards K. bridge. He had "
             "successfully avoided meeting his landlady on the staircase. His "
             "garret was under the roof of a high, five-storied house and was "
             "more like a cupboard than a room."},
    {"id": "cp_isol_fr", "work": "cp", "theme": "isolation", "lang": "fr",
     "text": "Au commencement de juillet, par une chaleur exceptionnelle, vers "
             "le soir, un jeune homme sortit de la chambrette qu'il "
             "sous-louait dans la ruelle S. et, lentement, d'un air irrésolu, "
             "se dirigea vers le pont K. Il avait réussi à éviter la rencontre "
             "de sa logeuse dans l'escalier. Sa chambrette se trouvait sous le "
             "toit même d'une haute maison de cinq étages et ressemblait "
             "plutôt à une armoire qu'à une pièce d'habitation."},

    # ═══════════════════════════════════════
    # ISOLATION — Notes (Underground Man)
    # ═══════════════════════════════════════
    {"id": "notes_isol_ru", "work": "notes", "theme": "isolation", "lang": "ru",
     "text": "Тогда мне было всего двадцать четыре года. Жизнь моя была уже "
             "и тогда угрюмая, беспорядочная и до одичалости одинокая. "
             "Я ни с кем не водился и даже избегал говорить и всё более и "
             "более забивался в свой угол. На службе, в канцелярии, я "
             "старался даже не глядеть ни на кого, и я очень хорошо замечал, "
             "что на меня товарищи не только смотрели как на чудака, но "
             "и — мне всё казалось это — смотрели как будто с каким-то "
             "омерзением."},
    {"id": "notes_isol_en", "work": "notes", "theme": "isolation", "lang": "en",
     "text": "At that time I was only twenty-four. My life was even then "
             "gloomy, ill-regulated, and as solitary as that of a savage. "
             "I made friends with no one and positively avoided talking, and "
             "buried myself more and more in my hole. At work in the office "
             "I never looked at any one, and I was perfectly well aware that "
             "my companions looked upon me, not only as a queer fellow, but "
             "even — I always fancied this — with a sort of loathing."},
    {"id": "notes_isol_fr", "work": "notes", "theme": "isolation", "lang": "fr",
     "text": "J'avais alors vingt-quatre ans. Ma vie était déjà sombre, "
             "désordonnée et solitaire jusqu'à la sauvagerie. Je ne "
             "fréquentais personne et évitais même de parler, m'enfonçant de "
             "plus en plus dans mon coin. Au bureau, je m'efforçais de ne "
             "regarder personne, et je remarquais très bien que mes collègues "
             "me considéraient non seulement comme un original, mais encore — "
             "du moins il me semblait — avec une sorte de dégoût."},

    # ═══════════════════════════════════════
    # ISOLATION — AK: Anna's despair (Part 7 Ch 31)
    # ═══════════════════════════════════════
    {"id": "ak_isol_ru", "work": "ak", "theme": "isolation", "lang": "ru",
     "text": "Да, на чем я остановилась? На том, что я не могу придумать "
             "положения, в котором жизнь не была бы мученьем, что все мы "
             "созданы затем, чтобы мучаться, и что мы все знаем это и все "
             "придумываем средства, как бы обмануть себя. А когда видишь "
             "правду, что же делать? Все неправда, все ложь, все обман, все "
             "зло! Когда поезд подошел к станции, Анна вышла в толпе других "
             "пассажиров и, как от прокаженных, сторонясь от них, остановилась "
             "на платформе, стараясь вспомнить, зачем она сюда приехала и что "
             "намерена была делать. Все, что ей казалось возможно прежде, "
             "теперь так трудно было сообразить, особенно в шумящей толпе всех "
             "этих безобразных людей, не оставлявших ее в покое."},
    {"id": "ak_isol_en", "work": "ak", "theme": "isolation", "lang": "en",
     "text": "Yes, what did I stop at? That I couldn't conceive a position in "
             "which life would not be a misery, that we are all created to be "
             "miserable, and that we all know it, and all invent means of "
             "deceiving each other. And when one sees the truth, what is one "
             "to do? It's all falsehood, all lying, all humbug, all cruelty! "
             "When the train came into the station, Anna got out into the "
             "crowd of passengers, and moving apart from them as if they were "
             "lepers, she stood on the platform, trying to think what she had "
             "come here for, and what she meant to do. Everything that had "
             "seemed to her possible before was now so difficult to consider, "
             "especially in this noisy crowd of hideous people who would not "
             "leave her alone."},
    {"id": "ak_isol_fr", "work": "ak", "theme": "isolation", "lang": "fr",
     "text": "Oui, où en étais-je restée? Que je ne pouvais concevoir une "
             "situation où la vie ne fût pas une souffrance, que nous sommes "
             "tous créés pour souffrir, que nous le savons tous, et que nous "
             "inventons tous des moyens de nous tromper les uns les autres. Et "
             "quand on voit la vérité, que faire? Tout est mensonge, tout est "
             "tromperie, tout est cruauté! Quand le train entra en gare, Anna "
             "descendit au milieu de la foule des voyageurs et, s'écartant "
             "d'eux comme s'ils étaient des lépreux, elle resta sur le quai, "
             "essayant de se rappeler pourquoi elle était venue et ce qu'elle "
             "avait l'intention de faire. Tout ce qui lui avait paru possible "
             "auparavant était maintenant si difficile à envisager, surtout au "
             "milieu de cette foule bruyante de gens hideux qui ne la "
             "laissaient pas en paix."},

    # ═══════════════════════════════════════
    # ISOLATION — BK: Ivan confronting himself (Book 11 Ch 10)
    # ═══════════════════════════════════════
    {"id": "bk_isol_ru", "work": "bk", "theme": "isolation", "lang": "ru",
     "text": "А он — это я, Алеша, я сам. Всё мое низкое, всё мое подлое и "
             "презренное! Да, я «романтик», он это подметил… хоть это и "
             "клевета. Он ужасно глуп, но он этим берет. Он хитер, животно "
             "хитер, он знал, чем взбесить меня. Он всё дразнил меня, что я "
             "в него верю, и тем заставил меня его слушать. Он надул меня, "
             "как мальчишку. Он мне, впрочем, сказал про меня много правды. "
             "Я бы никогда этого не сказал себе. Знаешь, Алеша, знаешь, — "
             "я бы очень желал, чтоб он в самом деле был он, а не я!"},
    {"id": "bk_isol_en", "work": "bk", "theme": "isolation", "lang": "en",
     "text": "And he is myself, Alyosha. All that's base in me, all that's "
             "mean and contemptible. Yes, I am a romantic. He guessed it ... "
             "though it's a libel. He is frightfully stupid; but it's to his "
             "advantage. He has cunning, animal cunning — he knew how to "
             "infuriate me. He kept taunting me with believing in him, and "
             "that was how he made me listen to him. He fooled me like a boy. "
             "He told me a great deal that was true about myself, though. I "
             "should never have owned it to myself. Do you know, Alyosha, I "
             "should be awfully glad to think that it was he and not I."},
    {"id": "bk_isol_fr", "work": "bk", "theme": "isolation", "lang": "fr",
     "text": "Et c'est moi, Aliocha, moi-même. Tout ce qu'il y a de bas en "
             "moi, tout ce qu'il y a de vil et de méprisable! Oui, je suis un "
             "«romantique», il l'a deviné... bien que ce soit une calomnie. Il "
             "est affreusement bête, mais c'est par là qu'il vous prend. Il "
             "est rusé, d'une ruse animale, il savait comment me mettre en "
             "fureur. Il me taquinait en disant que je croyais en lui, et "
             "c'est ainsi qu'il m'a forcé à l'écouter. Il m'a trompé comme un "
             "gamin. Il m'a dit pourtant beaucoup de vérités sur moi-même. Je "
             "ne me les serais jamais dites. Sais-tu, Aliocha, je voudrais "
             "bien que ce fût vraiment lui et non pas moi!"},

    # ═══════════════════════════════════════
    # FAMILY — AK Part 1 Ch 1 (Oblonsky)
    # ═══════════════════════════════════════
    {"id": "ak_fam_ru", "work": "ak", "theme": "family", "lang": "ru",
     "text": "Все счастливые семьи похожи друг на друга, каждая несчастливая "
             "семья несчастлива по-своему. Всё смешалось в доме Облонских. "
             "Жена узнала, что муж был в связи с бывшею в их доме "
             "француженкою-гувернанткой, и объявила мужу, что не может жить "
             "с ним в одном доме. Положение это продолжалось уже третий день "
             "и мучительно чувствовалось и самими супругами, и всеми членами "
             "семьи, и домочадцами."},
    {"id": "ak_fam_en", "work": "ak", "theme": "family", "lang": "en",
     "text": "Happy families are all alike; every unhappy family is unhappy "
             "in its own way. Everything was in confusion in the Oblonsky "
             "household. The wife had discovered that the husband was carrying "
             "on an intrigue with a French girl, who had been a governess in "
             "their family, and she had announced to her husband that she "
             "could not go on living in the same house with him. This position "
             "of affairs had now lasted three days, and not only the husband "
             "and wife themselves, but all the members of their family and "
             "household, were painfully conscious of it."},
    {"id": "ak_fam_fr", "work": "ak", "theme": "family", "lang": "fr",
     "text": "Toutes les familles heureuses se ressemblent; chaque famille "
             "malheureuse est malheureuse à sa façon. Tout était sens dessus "
             "dessous dans la maison Oblonski. La femme avait découvert que "
             "son mari entretenait une liaison avec leur ancienne gouvernante "
             "française, et avait déclaré à son mari qu'elle ne pouvait plus "
             "vivre avec lui sous le même toit. Cette situation durait depuis "
             "trois jours et était cruellement ressentie par les époux "
             "eux-mêmes, par tous les membres de la famille et par les "
             "domestiques."},

    # ═══════════════════════════════════════
    # FAMILY — BK opening (Karamazov)
    # ═══════════════════════════════════════
    {"id": "bk_fam_ru", "work": "bk", "theme": "family", "lang": "ru",
     "text": "Алексей Федорович Карамазов был третий сын помещика нашего "
             "уезда Федора Павловича Карамазова, столь известного в свое "
             "время по одному темному и трагическому делу, которое будет "
             "поведано мною в надлежащем месте. Теперь же скажу об этом "
             "помещике лишь то, что это был странный тип, довольно часто, "
             "однако, встречающийся, именно тип человека не только дрянного "
             "и развратного, но вместе с тем и бестолкового."},
    {"id": "bk_fam_en", "work": "bk", "theme": "family", "lang": "en",
     "text": "Alexey Fyodorovitch Karamazov was the third son of Fyodor "
             "Pavlovitch Karamazov, a landowner well known in our district "
             "in his own day, and still remembered among us owing to his "
             "gloomy and tragic death, which happened thirteen years ago, and "
             "which I shall describe in its proper place. For the present I "
             "will only say that this landowner was a strange type, yet one "
             "pretty frequently to be met with, a type abject and vicious and "
             "at the same time senseless."},
    {"id": "bk_fam_fr", "work": "bk", "theme": "family", "lang": "fr",
     "text": "Alexéi Fiodorovitch Karamazov était le troisième fils d'un "
             "propriétaire foncier de notre district, Fiodor Pavlovitch "
             "Karamazov, si célèbre en son temps à cause d'une affaire "
             "ténébreuse et tragique que je raconterai en son lieu. Pour le "
             "moment, je dirai seulement de ce propriétaire que c'était un "
             "type étrange, assez fréquent pourtant, le type d'un homme non "
             "seulement vil et débauché, mais encore absurde."},

    # ═══════════════════════════════════════
    # FAMILY — CP Part 1 Ch 2 (Marmeladov)
    # ═══════════════════════════════════════
    {"id": "cp_fam_ru", "work": "cp", "theme": "family", "lang": "ru",
     "text": "Милостивый государь, — начал он почти с торжественностью, — "
             "бедность не порок, это истина. Знаю я, что и пьянство не "
             "добродетель, и это тем паче. Но нищета, милостивый государь, "
             "нищета — порок-с. В бедности вы ещё сохраняете свое "
             "благородство врождённых чувств, в нищете же никогда и никто. "
             "За нищету даже и не палкой выгоняют, а метлой выметают из "
             "компании человеческой, чтобы тем оскорбительнее было."},
    {"id": "cp_fam_en", "work": "cp", "theme": "family", "lang": "en",
     "text": "'Honoured sir,' he began almost with solemnity, 'poverty is not "
             "a vice, that's a true saying. Yet I know too that drunkenness "
             "is not a virtue, and that that's even truer. But beggary, "
             "honoured sir, beggary is a vice. In poverty you may still "
             "retain your innate nobility of soul, but in beggary never — "
             "no one. For beggary a man is not chased out of human society "
             "with a stick, he is swept out with a broom, so as to make it "
             "as humiliating as possible.'"},
    {"id": "cp_fam_fr", "work": "cp", "theme": "family", "lang": "fr",
     "text": "«Monsieur, commença-t-il presque avec solennité, la pauvreté "
             "n'est pas un vice, c'est une vérité. Je sais aussi que "
             "l'ivrognerie n'est pas une vertu, et cela est encore plus vrai. "
             "Mais la misère, monsieur, la misère est un vice. Dans la "
             "pauvreté, vous conservez encore la noblesse de vos sentiments "
             "innés, mais dans la misère, jamais et personne. Dans la misère, "
             "on ne vous chasse pas à coups de bâton, on vous balaie du monde "
             "à coups de balai, pour que ce soit plus humiliant.»"},

    # ═══════════════════════════════════════
    # CONTROL — Kinder Surprise warning
    # ═══════════════════════════════════════
    {"id": "kinder_ru", "work": "kinder", "theme": "control", "lang": "ru",
     "text": "ВНИМАНИЕ! Прочитайте и сохраните: игрушку не предназначена для "
             "детей младше 3-х лет, мелкие детали могут быть проглочены или "
             "попасть в дыхательные пути. Рекомендуется наблюдение взрослых."},
    {"id": "kinder_en", "work": "kinder", "theme": "control", "lang": "en",
     "text": "WARNING: Read and keep. Toy not suitable for children under 3 "
             "years. Small parts may be swallowed or inhaled. Adult "
             "supervision recommended."},
    {"id": "kinder_fr", "work": "kinder", "theme": "control", "lang": "fr",
     "text": "ATTENTION: À lire et à conserver. Jouet ne convenant pas aux "
             "enfants de moins de 3 ans. Les petites pièces peuvent être "
             "avalées ou inhalées. Surveillance d'un adulte recommandée."},
]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("=" * 70)
    print("THEMATIC EXPERIMENT v4 — DEFINITIVE DESIGN")
    print("4 isolation works, 3 family works, Kinder control, 3 languages")
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
            print(f"[{datetime.now():%H:%M:%S}] Loaded {model_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    hidden_dim = model.config.hidden_size
    acts = {}
    for i, p in enumerate(PASSAGES):
        pid = p["id"]
        inputs = tokenizer(p["text"], return_tensors="pt").to(model.device)
        n_tok = inputs["input_ids"].shape[1]
        print(f"[{datetime.now():%H:%M:%S}] [{i+1}/{len(PASSAGES)}] {pid}: {n_tok} tok [{p['lang']}]")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        layer_means = np.zeros((len(outputs.hidden_states), hidden_dim), dtype=np.float32)
        for li, hs in enumerate(outputs.hidden_states):
            layer_means[li] = hs[0].float().mean(dim=0).cpu().numpy()
        acts[pid] = layer_means
        del outputs, inputs; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(OUTPUT_DIR, "activations.npz"), **acts)

    n_layers = acts["cp_isol_ru"].shape[0]
    meta = {p["id"]: p for p in PASSAGES}

    # ═══════════════════════════════════════
    # TEST 1: For each crossed work (AK, BK, CP), theme vs work
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 1: Per-work theme vs work (crossed works only)")
    print("="*70)

    crossed_works = ["ak", "bk", "cp"]

    for work in crossed_works:
        isol_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "isolation"]
        fam_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "family"]
        other_isol = [p["id"] for p in PASSAGES if p["work"] != work and p["theme"] == "isolation"]

        t_wins = 0; w_wins = 0
        print(f"\n  {work.upper()}: isolation vs family")
        for li in range(n_layers):
            # Work: same-work cross-theme cross-lingual
            w_sims = []
            for a in isol_ids:
                for b in fam_ids:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        w_sims.append(1 - cosine(acts[a][li], acts[b][li]))

            # Theme: same-theme cross-work cross-lingual
            t_sims = []
            for a in isol_ids:
                for b in other_isol:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        t_sims.append(1 - cosine(acts[a][li], acts[b][li]))

            t = np.mean(t_sims); w = np.mean(w_sims)
            if t > w: t_wins += 1
            else: w_wins += 1

            if li % 4 == 0:
                winner = "THEME" if t > w else "WORK"
                nm = "embed" if li == 0 else f"L{li}"
                print(f"    {nm:>8s}: theme={t:.4f} work={w:.4f} {winner} gap={t-w:+.4f}")

        print(f"    TOTAL: theme={t_wins} work={w_wins}")

    # ═══════════════════════════════════════
    # TEST 2: Kinder egg — does it cluster with anything?
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 2: Kinder egg control")
    print("="*70)

    peak = n_layers // 2
    nm = f"layer_{peak}"
    kinder_ids = [p["id"] for p in PASSAGES if p["work"] == "kinder"]
    lit_ids = [p["id"] for p in PASSAGES if p["work"] != "kinder"]

    # Kinder within-work cross-lingual similarity
    k_within = []
    for a, b in combinations(kinder_ids, 2):
        if meta[a]["lang"] != meta[b]["lang"]:
            k_within.append(1 - cosine(acts[a][peak], acts[b][peak]))

    # Kinder to literary cross-lingual
    k_lit = []
    for a in kinder_ids:
        for b in lit_ids:
            if meta[a]["lang"] != meta[b]["lang"]:
                k_lit.append(1 - cosine(acts[a][peak], acts[b][peak]))

    print(f"  At {nm}:")
    print(f"    Kinder within (cross-lingual): {np.mean(k_within):.4f}")
    print(f"    Kinder to literature (cross-lingual): {np.mean(k_lit):.4f}")
    print(f"    Gap: {np.mean(k_within) - np.mean(k_lit):+.4f}")

    # ═══════════════════════════════════════
    # TEST 3: Grand summary across all crossed works
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 3: Grand summary — all crossed works combined")
    print("="*70)

    grand_theme = 0; grand_work = 0
    for li in range(n_layers):
        all_t = []; all_w = []
        for work in crossed_works:
            isol_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "isolation"]
            fam_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "family"]
            other_isol = [p["id"] for p in PASSAGES if p["work"] != work and p["theme"] == "isolation"]

            for a in isol_ids:
                for b in fam_ids:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        all_w.append(1 - cosine(acts[a][li], acts[b][li]))
                for b in other_isol:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        all_t.append(1 - cosine(acts[a][li], acts[b][li]))

        t = np.mean(all_t); w = np.mean(all_w)
        if t > w: grand_theme += 1
        else: grand_work += 1

        if li % 4 == 0:
            winner = "THEME" if t > w else "WORK"
            nm = "embed" if li == 0 else f"L{li}"
            print(f"  {nm:>8s}: theme={t:.4f} work={w:.4f} {winner} gap={t-w:+.4f}")

    print(f"\n  GRAND TOTAL: theme={grand_theme}/{n_layers} work={grand_work}/{n_layers}")

    # ═══════════════════════════════════════
    # Plot
    # ═══════════════════════════════════════
    results_t = []; results_w = []
    for li in range(n_layers):
        all_t = []; all_w = []
        for work in crossed_works:
            isol_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "isolation"]
            fam_ids = [p["id"] for p in PASSAGES if p["work"] == work and p["theme"] == "family"]
            other_isol = [p["id"] for p in PASSAGES if p["work"] != work and p["theme"] == "isolation"]
            for a in isol_ids:
                for b in fam_ids:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        all_w.append(1 - cosine(acts[a][li], acts[b][li]))
                for b in other_isol:
                    if meta[a]["lang"] != meta[b]["lang"]:
                        all_t.append(1 - cosine(acts[a][li], acts[b][li]))
        results_t.append(np.mean(all_t))
        results_w.append(np.mean(all_w))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(n_layers), results_t, "o-", color="#2980b9", markersize=2,
            linewidth=1.5, label="Same theme, diff work (cross-lingual)")
    ax.plot(range(n_layers), results_w, "s-", color="#c0392b", markersize=2,
            linewidth=1.5, label="Same work, diff theme (cross-lingual)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Content Abstraction Test: Theme vs Work Identity\n"
                 "3 crossed works (AK, BK, CP) × 3 languages")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "theme_vs_work_v4.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({"theme_wins": grand_theme, "work_wins": grand_work,
                    "n_layers": n_layers}, f, indent=2)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
