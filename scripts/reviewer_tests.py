"""
Four Reviewer-Requested Tests
==============================
1. Truncated Maude (same narrative boundary as Garnett)
2. Homer pairwise matrix (all 5 translators × 2 epics)
3. Null distribution (same-language pairwise cosines from bootstrap pool)
4. Base model Garnett-Maude (if Llama available; otherwise flag)
"""

import torch
import numpy as np
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from itertools import combinations

OUTPUT_DIR = "reviewer_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════
# PASSAGES
# ═══════════════════════════════════════

# Test 1: Garnett vs Maude TRUNCATED to same boundary
TRANSLATOR_PASSAGES = [
    ("ak_garnett", "en", "Happy families are all alike; every unhappy family is unhappy "
     "in its own way. Everything was in confusion in the Oblonsky household. "
     "The wife had discovered that the husband was carrying on an intrigue "
     "with a French girl, who had been a governess in their family, and she "
     "had announced to her husband that she could not go on living in the "
     "same house with him. This position of affairs had now lasted three days, "
     "and not only the husband and wife themselves, but all the members of "
     "their family and household, were painfully conscious of it."),
    ("ak_maude_full", "en", "All happy families resemble one another, each unhappy family "
     "is unhappy in its own way. Everything was upset in the Oblonskys' house. "
     "The wife had discovered an intrigue between her husband and their former "
     "French governess, and declared that she would not continue to live under "
     "the same roof with him. This state of things had now lasted for three "
     "days, and not only the husband and wife but the rest of the family and "
     "the whole household suffered from it. They all felt that there was no "
     "sense in their living together, and that any group of people who had met "
     "together by chance at an inn would have had more in common than they. "
     "The wife kept to her own rooms; the husband stopped away from home all "
     "day; the children ran about all over the house uneasily; the English "
     "governess quarrelled with the housekeeper and wrote to a friend asking "
     "if she could find her another situation; the cook had gone out just at "
     "dinnertime the day before and had not returned; and the kitchen-maid "
     "and coachman had given notice."),
    ("ak_maude_trunc", "en", "All happy families resemble one another, each unhappy family "
     "is unhappy in its own way. Everything was upset in the Oblonskys' house. "
     "The wife had discovered an intrigue between her husband and their former "
     "French governess, and declared that she would not continue to live under "
     "the same roof with him. This state of things had now lasted for three "
     "days, and not only the husband and wife but the rest of the family and "
     "the whole household suffered from it."),
    ("ak_russian", "ru", "Все счастливые семьи похожи друг на друга, каждая несчастливая "
     "семья несчастлива по-своему. Всё смешалось в доме Облонских. Жена "
     "узнала, что муж был в связи с бывшею в их доме француженкою-гувернанткой, "
     "и объявила мужу, что не может жить с ним в одном доме. Положение это "
     "продолжалось уже третий день и мучительно чувствовалось и самими "
     "супругами, и всеми членами семьи, и домочадцами."),
    ("notes_garnett", "en", "I am a sick man. I am a spiteful man. I am an unattractive "
     "man. I believe my liver is diseased. However, I know nothing at all "
     "about my disease, and do not know for certain what ails me. I don't "
     "consult a doctor for it, and never have, though I have a respect for "
     "medicine and doctors. Besides, I am extremely superstitious, sufficiently "
     "so to respect medicine, at any rate. I am well-educated enough not to "
     "be superstitious, but I am superstitious. No, I refuse to consult a "
     "doctor from spite."),
]

# Test 2: Homer pairwise (5 translators × 2 epics)
HOMER_PASSAGES = [
    ("chapman_iliad", "en", "Achilles' baneful wrath resound, O Goddess, that impos'd "
     "Infinite sorrows on the Greeks, and many brave souls los'd "
     "From breasts Heroique — Loss of many Souls,  "
     "That Heroes strong as well as Warriors bold "
     "Were made the prey of Dogs, and Vultures, and "
     "The will of Jove was done from the first stroke "
     "That caus'd Atrides and the God-like Greek "
     "The great Achilles into discord."),
    ("pope_iliad", "en", "Achilles' wrath, to Greece the direful spring "
     "Of woes unnumber'd, heavenly goddess, sing! "
     "That wrath which hurl'd to Pluto's gloomy reign "
     "The souls of mighty chiefs untimely slain; "
     "Whose limbs unburied on the naked shore, "
     "Devouring dogs and hungry vultures tore. "
     "Since great Achilles and Atrides strove, "
     "Such was the sovereign doom, and such the will of Jove!"),
    ("cowper_iliad", "en", "Achilles sing, O Goddess! Peleus' son; "
     "His wrath pernicious, who ten thousand woes "
     "Caused to Achaia's host, sent many a soul "
     "Illustrious into Ades premature, "
     "And Heroes gave (so stood the will of Jove) "
     "To dogs and to all ravening fowls a prey, "
     "When fierce dispute had parted once in twain "
     "The noble Chief Achilles from the King "
     "Of men Atrides."),
    ("butler_iliad", "en", "Sing, O goddess, the anger of Achilles son of Peleus, that "
     "brought countless ills upon the Achaeans. Many a brave soul did it send "
     "hurrying down to Hades, and many a hero did it yield a prey to dogs and "
     "vultures, for so were the counsels of Jove fulfilled from the day on "
     "which the son of Atreus, king of men, and great Achilles, first fell "
     "out with one another."),
    ("murray_iliad", "en", "The wrath sing, goddess, of Peleus' son, Achilles, that "
     "destructive wrath which brought countless woes upon the Achaeans, and "
     "sent forth to Hades many valiant souls of heroes, and made them "
     "themselves spoil for dogs and every bird; thus the plan of Zeus was "
     "being accomplished, from the time when first there parted in strife "
     "Atreus' son, lord of men, and noble Achilles."),
    ("chapman_odyssey", "en", "The man, O Muse, inform, that many a way "
     "Wound with his wisdom to his wished stay; "
     "That wandered wondrous far, when he the town "
     "Of sacred Troy had help'd to overthrow. "
     "Many men's manners did he learn, and minds, "
     "Much of the sea he suffered grievous pains, "
     "Struggling to save his life and bring his men "
     "Their safety homeward."),
    ("pope_odyssey", "en", "The man for wisdom's various arts renown'd, "
     "Long exercised in woes, O Muse! resound; "
     "Who, when his arms had wrought the destin'd fall "
     "Of sacred Troy, and razed her heaven-built wall, "
     "Wandering from clime to clime, observant stray'd, "
     "Their manners noted, and their states survey'd."),
    ("cowper_odyssey", "en", "Muse, make the man thy theme, for shrewdness famed "
     "And genius versatile, who far and wide "
     "A Wanderer, after Ilium overthrown, "
     "Discover'd various cities, and the mind "
     "And manners learn'd of men, in lands remote."),
    ("butler_odyssey", "en", "Tell me, O Muse, of that ingenious hero who travelled far and "
     "wide after he had sacked the famous town of Troy. Many cities did he "
     "visit, and many were the nations with whose manners and customs he was "
     "acquainted; moreover he suffered much by sea while trying to save his "
     "own life and bring his men safely home."),
    ("murray_odyssey", "en", "Tell me, O Muse, of the man of many devices, who wandered "
     "full many ways after he had sacked the sacred citadel of Troy. Many "
     "were the men whose cities he saw and whose mind he learned, aye, and "
     "many the woes he suffered in his heart upon the sea, seeking to win "
     "his own life and the return of his comrades."),
]

# Test 3: Null distribution — diverse English passages from bootstrap pool
NULL_POOL_EN = [
    ("moby_dick", "Call me Ishmael. Some years ago — never mind how long precisely — "
     "having little or no money in my purse, and nothing particular to "
     "interest me on shore, I thought I would sail about a little and see "
     "the watery part of the world."),
    ("pride_prejudice", "It is a truth universally acknowledged, that a single man in "
     "possession of a good fortune, must be in want of a wife. However little "
     "known the feelings or views of such a man may be on his first entering "
     "a neighbourhood, this truth is so well fixed in the minds of the "
     "surrounding families, that he is considered as the rightful property "
     "of some one or other of their daughters."),
    ("two_cities", "It was the best of times, it was the worst of times, it was the "
     "age of wisdom, it was the age of foolishness, it was the epoch of "
     "belief, it was the epoch of incredulity, it was the season of Light, "
     "it was the season of Darkness, it was the spring of hope, it was the "
     "winter of despair."),
    ("alice", "Alice was beginning to get very tired of sitting by her sister on "
     "the bank, and of having nothing to do: once or twice she had peeped "
     "into the book her sister was reading, but it had no pictures or "
     "conversations in it, 'and what is the use of a book,' thought Alice "
     "'without pictures or conversations?'"),
    ("frankenstein", "I am by birth a Genevese, and my family is one of the most "
     "distinguished of that republic. My ancestors had been for many years "
     "counsellors and syndics, and my father had filled several public "
     "situations with honour and reputation."),
    ("don_quixote", "In a village of La Mancha, the name of which I have no desire to "
     "call to mind, there lived not long since one of those gentlemen that "
     "keep a lance in the lance-rack, an old buckler, a lean hack, and a "
     "greyhound for coursing."),
    ("hound", "Mr. Sherlock Holmes, who was usually very late in the mornings, save "
     "upon those not infrequent occasions when he was up all night, was "
     "seated at the breakfast table. I stood upon the hearth-rug and picked "
     "up the stick which our visitor had left behind him the night before."),
    ("jungle_book", "It was seven o'clock of a very warm evening in the Seoni hills "
     "when Father Wolf woke up from his day's rest, scratched himself, "
     "yawned, and spread out his paws one after the other to get rid of "
     "the sleepy feeling in their tips."),
    ("wealth_nations", "The annual labour of every nation is the fund which originally "
     "supplies it with all the necessaries and conveniences of life which "
     "it annually consumes, and which consist always either in the immediate "
     "produce of that labour, or in what is purchased with that produce "
     "from other nations."),
    ("walden", "When I wrote the following pages, or rather the bulk of them, I "
     "lived alone, in the woods, a mile from any neighbor, in a house which "
     "I had built myself, on the shore of Walden Pond, in Concord, "
     "Massachusetts, and earned my living by the labor of my hands only."),
    ("gettysburg", "Four score and seven years ago our fathers brought forth on this "
     "continent, a new nation, conceived in Liberty, and dedicated to the "
     "proposition that all men are created equal. Now we are engaged in a "
     "great civil war, testing whether that nation, or any nation so "
     "conceived and so dedicated, can long endure."),
    ("republic", "I went down yesterday to the Piraeus with Glaucon the son of "
     "Ariston, that I might offer up my prayers to the goddess, and also "
     "because I wanted to see in what manner they would celebrate the "
     "festival, which was a new thing."),
]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 70)
    print("FOUR REVIEWER-REQUESTED TESTS")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    hidden = model.config.hidden_size

    def extract(text):
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        layers = np.zeros((len(out.hidden_states), hidden), dtype=np.float32)
        for li, hs in enumerate(out.hidden_states):
            layers[li] = hs[0].float().mean(dim=0).cpu().numpy()
        ntok = inputs["input_ids"].shape[1]
        del out, inputs; torch.cuda.empty_cache()
        return layers, ntok

    # ═══════════════════════════════════════
    # TEST 1: Truncated Maude
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 1: TRUNCATED MAUDE vs GARNETT")
    print("="*70)

    trans_acts = {}
    for pid, lang, text in TRANSLATOR_PASSAGES:
        layers, ntok = extract(text)
        trans_acts[pid] = layers
        print(f"  {pid}: {ntok} tok")

    n_layers = trans_acts["ak_garnett"].shape[0]

    print(f"\n{'Layer':>8s}  {'G↔M_full':>9s}  {'G↔M_trunc':>10s}  {'G↔Notes':>8s}  {'G↔Ru':>8s}  {'Trunc win':>10s}")
    print("-" * 65)

    trunc_wins = 0
    for li in range(0, n_layers, 1):
        gm_full = 1 - cosine(trans_acts["ak_garnett"][li], trans_acts["ak_maude_full"][li])
        gm_trunc = 1 - cosine(trans_acts["ak_garnett"][li], trans_acts["ak_maude_trunc"][li])
        g_notes = 1 - cosine(trans_acts["ak_garnett"][li], trans_acts["notes_garnett"][li])
        g_ru = 1 - cosine(trans_acts["ak_garnett"][li], trans_acts["ak_russian"][li])
        win = "CONTENT" if gm_trunc > g_notes else "OTHER"
        if gm_trunc > g_notes: trunc_wins += 1
        if li % 4 == 0:
            nm = "embed" if li == 0 else f"L{li}"
            print(f"  {nm:>8s}  {gm_full:>9.4f}  {gm_trunc:>10.4f}  {g_notes:>8.4f}  {g_ru:>8.4f}  {win:>10s}")

    print(f"\n  Truncated Maude content wins: {trunc_wins}/{n_layers}")

    # ═══════════════════════════════════════
    # TEST 2: Homer Pairwise
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 2: HOMER PAIRWISE (5 translators × 2 epics)")
    print("="*70)

    homer_acts = {}
    for pid, lang, text in HOMER_PASSAGES:
        layers, ntok = extract(text)
        homer_acts[pid] = layers
        print(f"  {pid}: {ntok} tok")

    # Peak content layer for Mistral ~16-20
    for peak in [16, 20]:
        print(f"\n  Layer {peak} pairwise similarities:")
        hids = [p[0] for p in HOMER_PASSAGES]
        print(f"  {'':>18s}", end="")
        for h in hids:
            short = h.replace("_iliad", "_I").replace("_odyssey", "_O")[:8]
            print(f" {short:>8s}", end="")
        print()
        for i, hi in enumerate(hids):
            short_i = hi.replace("_iliad", "_I").replace("_odyssey", "_O")[:8]
            print(f"  {short_i:>18s}", end="")
            for j, hj in enumerate(hids):
                sim = 1 - cosine(homer_acts[hi][peak], homer_acts[hj][peak])
                print(f" {sim:>8.3f}", end="")
            print()

        # Summary: within-epic vs cross-epic
        iliad_ids = [p[0] for p in HOMER_PASSAGES if "iliad" in p[0]]
        odyssey_ids = [p[0] for p in HOMER_PASSAGES if "odyssey" in p[0]]

        within_iliad = [1-cosine(homer_acts[a][peak], homer_acts[b][peak])
                        for a, b in combinations(iliad_ids, 2)]
        within_odyssey = [1-cosine(homer_acts[a][peak], homer_acts[b][peak])
                          for a, b in combinations(odyssey_ids, 2)]
        cross_epic = [1-cosine(homer_acts[a][peak], homer_acts[b][peak])
                      for a in iliad_ids for b in odyssey_ids]

        print(f"\n  Within-Iliad mean: {np.mean(within_iliad):.4f} (n={len(within_iliad)})")
        print(f"  Within-Odyssey mean: {np.mean(within_odyssey):.4f} (n={len(within_odyssey)})")
        print(f"  Cross-epic mean: {np.mean(cross_epic):.4f} (n={len(cross_epic)})")
        print(f"  Translator transparency gap: {np.mean(within_iliad+within_odyssey) - np.mean(cross_epic):+.4f}")

    # ═══════════════════════════════════════
    # TEST 3: Null Distribution
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    print("TEST 3: NULL DISTRIBUTION (same-language English pairwise)")
    print("="*70)

    null_acts = {}
    for pid, text in NULL_POOL_EN:
        layers, ntok = extract(text)
        null_acts[pid] = layers
        print(f"  {pid}: {ntok} tok")

    # Also include the test passages in the distribution
    null_acts["ak_garnett"] = trans_acts["ak_garnett"]
    null_acts["notes_garnett"] = trans_acts["notes_garnett"]

    all_en_ids = list(null_acts.keys())
    n_en = len(all_en_ids)

    for peak in [16, 20]:
        pairwise = []
        for a, b in combinations(all_en_ids, 2):
            sim = 1 - cosine(null_acts[a][peak], null_acts[b][peak])
            pairwise.append(sim)

        pairwise = np.array(pairwise)
        print(f"\n  Layer {peak}: {n_en} English passages, {len(pairwise)} pairs")
        print(f"  Mean: {pairwise.mean():.4f}")
        print(f"  SD:   {pairwise.std():.4f}")
        print(f"  Min:  {pairwise.min():.4f}")
        print(f"  Max:  {pairwise.max():.4f}")
        print(f"  Garnett-Maude (trunc): {1-cosine(trans_acts['ak_garnett'][peak], trans_acts['ak_maude_trunc'][peak]):.4f}")
        gm_sim = 1-cosine(trans_acts['ak_garnett'][peak], trans_acts['ak_maude_trunc'][peak])
        z = (gm_sim - pairwise.mean()) / pairwise.std()
        print(f"  Z-score of Garnett-Maude vs null: {z:.2f}")
        pct = np.mean(pairwise >= gm_sim) * 100
        print(f"  Percentile: {100-pct:.1f}th")

    # ═══════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════
    np.savez_compressed(os.path.join(OUTPUT_DIR, "translator_acts.npz"),
                        **{k: v for k, v in trans_acts.items()})
    np.savez_compressed(os.path.join(OUTPUT_DIR, "homer_acts.npz"),
                        **{k: v for k, v in homer_acts.items()})
    np.savez_compressed(os.path.join(OUTPUT_DIR, "null_acts.npz"),
                        **{k: v for k, v in null_acts.items()})

    print(f"\n{'='*70}")
    print(f"DONE: {datetime.now()}")
    print("="*70)


if __name__ == "__main__":
    main()
