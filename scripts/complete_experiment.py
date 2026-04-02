"""
COMPLETE EXPERIMENT: Scaled + Causal Tests
============================================
Cell 1: !pip install transformers==4.44.0 accelerate==0.33.0 bitsandbytes==0.43.3 huggingface_hub -q
         from huggingface_hub import login
         login(token="YOUR_TOKEN")

Cell 2: Paste everything below.
"""

import torch
import numpy as np
import os
from datetime import datetime
from math import comb

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ═══════════════════════════════════════
# PASSAGES
# ═══════════════════════════════════════

AK = [
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

NOTES = [
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

RU = [
    ("ak_ru_1", "ak", "Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива по-своему. Всё смешалось в доме Облонских. Жена узнала, что муж был в связи с бывшею в их доме француженкою-гувернанткой, и объявила мужу, что не может жить с ним в одном доме."),
    ("ak_ru_2", "ak", "В конце зимы, у Щербацких, происходил консилиум, долженствовавший решить, в каком положении находится здоровье Кити и что нужно предпринять для восстановления ее ослабевающих сил. Она была больна, и с приближением весны здоровье ее становилось хуже."),
    ("ak_ru_3", "ak", "Левин был женат третий месяц. Он был счастлив, но совсем не так, как ожидал. На каждом шагу он находил разочарование в прежних мечтах и новое, неожиданное очарование."),
    ("notes_ru_1", "notes", "Я человек больной... Я злой человек. Непривлекательный я человек. Я думаю, что у меня болит печень. Впрочем, я ни шиша не смыслю в моей болезни и не знаю наверно, что у меня болит."),
    ("notes_ru_2", "notes", "Я не только злым, но даже и ничем не сумел сделаться: ни злым, ни добрым, ни подлецом, ни честным, ни героем, ни насекомым. Теперь же доживаю в своем углу, дразня себя злобной и ни к чему не служащей утехой."),
    ("notes_ru_3", "notes", "Тогда мне было всего двадцать четыре года. Жизнь моя была уже и тогда угрюмая, беспорядочная и до одичалости одинокая. Я ни с кем не водился и даже избегал говорить и всё более и более забивался в свой угол."),
]

# ═══════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 70)
print(f"COMPLETE EXPERIMENT: Scaled + Causal | {datetime.now()}")
print("=" * 70)

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct",
                                              quantization_config=bnb, device_map="auto",
                                              torch_dtype=torch.float16)
model.eval()
hidden = model.config.hidden_size
n_model_layers = len(model.model.layers)
print(f"Loaded Llama 70B. Hidden: {hidden}, Layers: {n_model_layers}")

def get_acts(text):
    inputs = tok(text, return_tensors="pt").to(model.device)
    ntok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    layers = np.zeros((len(out.hidden_states), hidden), dtype=np.float32)
    for li, hs in enumerate(out.hidden_states):
        layers[li] = hs[0].float().mean(dim=0).cpu().numpy()
    del out, inputs; torch.cuda.empty_cache()
    return layers, ntok

# ═══════════════════════════════════════
# PART 1: EXTRACT ALL ACTIVATIONS
# ═══════════════════════════════════════

print("\n--- AK English (10) ---")
ak_acts = []
for i, p in enumerate(AK):
    a, n = get_acts(p); ak_acts.append(a)
    print(f"  AK_{i+1}: {n} tok")

print("--- Notes English (10) ---")
notes_acts = []
for i, p in enumerate(NOTES):
    a, n = get_acts(p); notes_acts.append(a)
    print(f"  N_{i+1}: {n} tok")

print("--- Russian (6) ---")
ru_data = []
for pid, work, text in RU:
    a, n = get_acts(text); ru_data.append((pid, work, a))
    print(f"  {pid}: {n} tok")

n_layers = ak_acts[0].shape[0]

# Build centroids
ak_cent = {}; notes_cent = {}
for li in range(n_layers):
    ak_cent[li] = np.mean([a[li] for a in ak_acts], axis=0)
    notes_cent[li] = np.mean([a[li] for a in notes_acts], axis=0)

# ═══════════════════════════════════════
# PART 2: SCALED EXPERIMENT TESTS
# ═══════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 1: ENGLISH CLUSTERING (10 AK vs 10 Notes)")
print("="*70)
print(f"{'Layer':>8s}  {'W-AK':>8s}  {'W-Notes':>8s}  {'Cross':>8s}  {'Gap':>8s}")
for li in range(0, n_layers, 5):
    w_ak = [cos_sim(ak_acts[i][li], ak_acts[j][li]) for i in range(10) for j in range(i+1,10)]
    w_n = [cos_sim(notes_acts[i][li], notes_acts[j][li]) for i in range(10) for j in range(i+1,10)]
    cross = [cos_sim(ak_acts[i][li], notes_acts[j][li]) for i in range(10) for j in range(10)]
    within = np.mean(w_ak + w_n); cross_m = np.mean(cross)
    nm = "embed" if li == 0 else f"L{li}"
    print(f"  {nm:>8s}  {np.mean(w_ak):>8.4f}  {np.mean(w_n):>8.4f}  {cross_m:>8.4f}  {within-cross_m:>+8.4f}")

print(f"\n{'='*70}")
print("TEST 2: LEAVE-ONE-OUT (n=20)")
print("="*70)
all_en = [("ak",a) for a in ak_acts] + [("notes",a) for a in notes_acts]
for li in range(0, n_layers, 5):
    correct = 0
    for idx in range(20):
        w, act = all_en[idx]
        ak_c = np.mean([a[li] for ww,a in all_en[:idx]+all_en[idx+1:] if ww=="ak"], axis=0)
        n_c = np.mean([a[li] for ww,a in all_en[:idx]+all_en[idx+1:] if ww=="notes"], axis=0)
        pred = "ak" if cos_sim(act[li], ak_c) > cos_sim(act[li], n_c) else "notes"
        if pred == w: correct += 1
    p = sum(comb(20,k)*0.5**20 for k in range(correct,21))
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
    nm = "embed" if li == 0 else f"L{li}"
    print(f"  {nm:>8s}  {correct}/20  p={p:.8f} {sig}")

print(f"\n{'='*70}")
print("TEST 3: CROSS-LINGUAL (6 Russian probes)")
print("="*70)
for li in range(0, n_layers, 5):
    correct = 0
    for pid, work, acts in ru_data:
        pred = "ak" if cos_sim(acts[li], ak_cent[li]) > cos_sim(acts[li], notes_cent[li]) else "notes"
        if pred == work: correct += 1
    nm = "embed" if li == 0 else f"L{li}"
    print(f"  {nm:>8s}  {correct}/6")

# ═══════════════════════════════════════
# PART 3: CAUSAL TEST — Activation Patching
# ═══════════════════════════════════════

print(f"\n{'='*70}")
print("CAUSAL TEST 1: ACTIVATION PATCHING")
print("Patch AK_ru_1 with Notes_ru_1 at peak content layer")
print("="*70)

source_text = RU[0][2]  # ak_ru_1
donor_text = RU[3][2]   # notes_ru_1
patch_layer = 40

# Get donor hidden state at patch layer
donor_state = {}
def capture_donor(module, input, output):
    donor_state['h'] = output[0].detach().clone()
h = model.model.layers[patch_layer].register_forward_hook(capture_donor)
d_in = tok(donor_text, return_tensors="pt").to(model.device)
with torch.no_grad(): model(**d_in)
h.remove(); del d_in
print(f"Captured Notes_ru hidden at L{patch_layer}: {donor_state['h'].shape}")

# Get UNPATCHED source activations at all post-patch layers
unpatched = {}
def make_cap(li):
    def hook(module, input, output):
        unpatched[li] = output[0].detach().float().cpu().mean(dim=1).squeeze().numpy()
    return hook
hooks = [model.model.layers[li].register_forward_hook(make_cap(li)) for li in range(patch_layer+1, n_model_layers)]
s_in = tok(source_text, return_tensors="pt").to(model.device)
with torch.no_grad(): model(**s_in)
for hk in hooks: hk.remove()

# Get PATCHED source activations
patched = {}
def make_cap2(li):
    def hook(module, input, output):
        patched[li] = output[0].detach().float().cpu().mean(dim=1).squeeze().numpy()
    return hook
def patch_fn(module, input, output):
    dp = donor_state['h'].float().mean(dim=1, keepdim=True).expand_as(output[0])
    return (dp.to(output[0].dtype),) + output[1:]

hooks2 = [model.model.layers[li].register_forward_hook(make_cap2(li)) for li in range(patch_layer+1, n_model_layers)]
ph = model.model.layers[patch_layer].register_forward_hook(patch_fn)
with torch.no_grad(): model(**s_in)
ph.remove()
for hk in hooks2: hk.remove()
del s_in; torch.cuda.empty_cache()

print(f"\n{'Layer':>8s}  {'Unpatch→AK':>11s}  {'Unpatch→N':>10s}  {'Patch→AK':>9s}  {'Patch→N':>8s}  {'Shift':>10s}")
print("-" * 60)
for li in sorted(patched.keys()):
    if li % 5 != 0: continue
    # Use li+1 for centroid index because hidden_states[0] = embed
    ci = li + 1
    if ci >= n_layers: continue
    u_ak = cos_sim(unpatched[li], ak_cent[ci])
    u_n = cos_sim(unpatched[li], notes_cent[ci])
    p_ak = cos_sim(patched[li], ak_cent[ci])
    p_n = cos_sim(patched[li], notes_cent[ci])
    shift = (p_n - p_ak) - (u_n - u_ak)
    direction = "→Notes" if shift > 0 else "→AK"
    print(f"  L{li:>5d}  {u_ak:>11.4f}  {u_n:>10.4f}  {p_ak:>9.4f}  {p_n:>8.4f}  {shift:>+8.4f} {direction}")

# ═══════════════════════════════════════
# PART 4: CAUSAL TEST — Steering with content direction
# ═══════════════════════════════════════

print(f"\n{'='*70}")
print("CAUSAL TEST 2: STEERING")
print("Add AK direction to Notes_ru_1 at layer 40")
print("="*70)

# AK direction = AK centroid - Notes centroid at peak layer
steer_layer = 40
ak_dir = ak_cent[steer_layer+1] - notes_cent[steer_layer+1]  # +1 for hidden_state indexing
ak_dir_tensor = torch.tensor(ak_dir, dtype=torch.float16).to(model.device)
alpha = 5.0  # steering strength

steer_text = RU[3][2]  # notes_ru_1

# Unsteered
unsteered = {}
def make_cap3(li):
    def hook(module, input, output):
        unsteered[li] = output[0].detach().float().cpu().mean(dim=1).squeeze().numpy()
    return hook
hooks3 = [model.model.layers[li].register_forward_hook(make_cap3(li)) for li in range(steer_layer+1, n_model_layers)]
st_in = tok(steer_text, return_tensors="pt").to(model.device)
with torch.no_grad(): model(**st_in)
for hk in hooks3: hk.remove()

# Steered
steered = {}
def make_cap4(li):
    def hook(module, input, output):
        steered[li] = output[0].detach().float().cpu().mean(dim=1).squeeze().numpy()
    return hook
def steer_fn(module, input, output):
    h = output[0].float()
    h = h + alpha * ak_dir_tensor.unsqueeze(0).unsqueeze(0)
    return (h.to(output[0].dtype),) + output[1:]

hooks4 = [model.model.layers[li].register_forward_hook(make_cap4(li)) for li in range(steer_layer+1, n_model_layers)]
sh = model.model.layers[steer_layer].register_forward_hook(steer_fn)
with torch.no_grad(): model(**st_in)
sh.remove()
for hk in hooks4: hk.remove()
del st_in; torch.cuda.empty_cache()

print(f"\nSteering: Notes_ru + {alpha}×(AK_centroid - Notes_centroid) at L{steer_layer}")
print(f"\n{'Layer':>8s}  {'Unsteer→AK':>11s}  {'Unsteer→N':>10s}  {'Steer→AK':>9s}  {'Steer→N':>8s}  {'Shift':>10s}")
print("-" * 60)
for li in sorted(steered.keys()):
    if li % 5 != 0: continue
    ci = li + 1
    if ci >= n_layers: continue
    u_ak = cos_sim(unsteered[li], ak_cent[ci])
    u_n = cos_sim(unsteered[li], notes_cent[ci])
    s_ak = cos_sim(steered[li], ak_cent[ci])
    s_n = cos_sim(steered[li], notes_cent[ci])
    shift = (s_ak - s_n) - (u_ak - u_n)
    direction = "→AK" if shift > 0 else "→Notes"
    print(f"  L{li:>5d}  {u_ak:>11.4f}  {u_n:>10.4f}  {s_ak:>9.4f}  {s_n:>8.4f}  {shift:>+8.4f} {direction}")

# ═══════════════════════════════════════
# SAVE
# ═══════════════════════════════════════

os.makedirs("results", exist_ok=True)
np.savez_compressed("results/scaled_causal.npz",
    **{f"ak_en_{i}": ak_acts[i] for i in range(10)},
    **{f"notes_en_{i}": notes_acts[i] for i in range(10)},
    **{pid: a for pid, w, a in ru_data},
)
print(f"\n{'='*70}")
print(f"ALL DONE: {datetime.now()}")
print("="*70)
