# PunQuizCN — Datasheet

Full documentation of the **PunQuizCN** corpus and benchmark, following the datasheet
template of Gebru et al., *Datasheets for Datasets* (CACM 64(12), 2021).
The paper-level summary of this document is the Datasheet appendix of the submission;
this file and `README.md` are the canonical, versioned documentation.

- Corpus version: **1.0** (every released file carries a version string; the corpus
  itself is hosted separately from this code repository — see `README.md` → Data)
- Licence: annotations, metadata, prompts and code **CC BY-NC 4.0**; panel images
  remain © their original authors (see [Distribution](#6-distribution))
- Maintainer: see `README.md` → Contact

---

## 1. Motivation

**For what purpose was the dataset created?**
To measure whether vision-language models can solve *Chinese two-image homophone rebuses*
(两张图猜谐音梗): a puzzle shows a hint card whose printed sentence names an object, and a
guess card whose scene must be read jointly with the first so that the two together sound
like a different word or phrase (the answer). Solving requires scene reading, phonological
awareness, and cultural knowledge at once — a combination that existing benchmarks do not
isolate, because prior Chinese pun resources pair a *single* image with a caption.

**Who created it and who funded it?**
The authors of the PunQuizCN paper (see `README.md`). No commercial funding was involved.

**Any other comments?**
The corpus is a *benchmark and diagnostic* resource, not a training corpus, and its size is
deliberately bounded by what human annotators could verify rather than by what could be
scraped.

## 2. Composition

**What do the instances represent?**
One instance = one puzzle. Each instance contains:

| field | meaning |
|---|---|
| `id` | stable item id |
| `account`, `post`, `from`, `created_at` | source metadata (WeChat official account, post, collection time) |
| `hint_img`, `guess_img` | the two panel images of the puzzle |
| `hint_text` | the hint card's printed sentence (usually `这是X`, "this is X") |
| `guess_text` | the free-form description of the guess card, written by the annotator |
| `answer` | the canonical answer (typically a 2–4 character word, idiom, name or phrase) |
| `align` | per-syllable alignment of the answer against `hint_text` + `guess_text` |
| `group_id`, `is_rep` | puzzle identity group and representative flag (see de-duplication) |
| `split` | group-disjoint 80/10/10 split over puzzle identity: `train` / `dev` / `test` |
| `created_by` | pseudonymous annotator token (never a name or contact detail) |

**How many instances are there?**
11,109 puzzles were annotated. Grouping by puzzle identity leaves 10,223 groups; 64 items
were then removed by quality screening, leaving **10,000 valid items**
(`train` 8,026 / `dev` 979 / `test` 995).

**What does each instance consist of?**
Two images plus text fields. The images are the two panels of the original puzzle, kept as
separate files per item (`<id>__hint.jpg`, `<id>__guess.jpg`). At inference both are downscaled so
that the long side is at most 640 px and re-encoded as JPEG (quality 85) before being attached to
the prompt, hint card first.

**Is there a label or target associated with each instance?**
Yes: `answer` (the canonical answer) and, for the evaluation tasks, the judging tiers derived
from it. Answer statistics: 9,849 items (98.5%) have 2–4 characters, 4,387 answer strings are
unique, 197 items contain Latin letters and 312 contain digits.

**Is any information missing?**
Descriptions (`guess_text`) are free-form and therefore incomplete by construction: an
annotator records what they notice. Any "homophone coverage" computed from descriptions is a
lower bound, which is why descriptions are used for ablations only, never as a substitute for
the images.

**Are relationships between instances made explicit?**
Yes: `group_id` links items that are the same puzzle reposted by different accounts, and
`is_rep` marks the representative item of each group.

**Are there recommended data splits?**
Yes: `split` is a **group-disjoint** 80/10/10 split over puzzle identity
(`train` 8,026 / `dev` 979 / `test` 995), computed as a stable hash of `group_id`
(`md5(group_id) % 10`: 0–7 → `train`, 8 → `dev`, 9 → `test`), so a whole
de-duplication cluster — every repost of one puzzle — always lands in the same
split. It is provided for reproducibility and as a leakage-safe default for
downstream fine-tuning. It is *not* used by the benchmark results reported in the
paper: those are zero-shot over all 10,000 items, and leakage there is controlled
by publication timestamp instead (the whole collection window postdates every
evaluated model's pre-training cutoff, so the corpus is effectively held out as a
whole). The names `train`/`dev`/`test` are a convention — the split is random, not
temporal.

**Are there errors, sources of noise, or redundancies?**
De-duplication is lexical, not semantic (image-pair identity, description equality, and
character n-gram similarity with Levenshtein verification), so puzzles that pose the same pun
with different drawings can survive as separate items. One account contributes 36% of the
items; dialect and region-specific puns are under-represented; readings are standard Mandarin.

**Is the dataset self-contained?**
Yes. Images are included in the release; no external downloads are needed. Source posts are
recorded as metadata for attribution only.

## 3. Collection Process

**How was the data acquired?**
From 38 public WeChat official accounts (`mp.weixin.qq.com`) that publish two-image homophone
rebuses. 3,848 posts published between 2025-09 and 2026-07 were read through their public web
pages; the puzzle panels and their printed text were extracted together with the post URL, the
account name and the publication date. No private messages, account credentials, follower data
or other user-level data were accessed.

**Who collected it and how were they compensated?**
Collection and annotation were carried out by the authors and by student research assistants
(see §4). Annotators were paid per item at the rate standard for student assistants at the
host institutions.

**Over what timeframe?**
Posts: 2025-09 → 2026-07. Annotation: 2026-07-20 → 2026-08-25.

**Were ethical review processes conducted?**
No ethics review was required or sought: the work involves no intervention on human
participants, no sensitive or demographic data from annotators, and no personal data from third
parties beyond what appears in already-published puzzle images and text. Annotators were
informed of the intended use (a benchmark released for non-commercial research), that their
identifiers would be pseudonymised in the release, and that they could stop at any time; all
consented.

**Does the dataset relate to people?**
Only indirectly: puzzle images may depict people as illustrations (part of the published
puzzle), and some answers are person or place names. The dataset contains no data *about*
identifiable individuals, and no annotator contact details.

## 4. Preprocessing, Cleaning, Labelling

**Was any preprocessing/cleaning/labelling done?**
Yes.

1. **Grouping and de-duplication.** Items are grouped by puzzle identity; within a group, one
   item is kept as the representative. Tiers: exact image-pair identity → exact equality of the
   two descriptions → character n-gram similarity with Levenshtein verification.
2. **Quality screening.** 64 items failed screening (missing/illegible panels, non-puzzle
   content, unresolvable answers).
3. **Alignment.** Each answer is decomposed into syllables and aligned against the two card
   texts with a documented phonological convention (initial/final/tone, with a small override
   and variant table); near-homophony is decided by toneless syllable match with tone
   differences recorded rather than discarded.
4. **Judging.** Model answers are compared to the canonical answer under three comparison modes
   and assigned a tier, which is what the reported strict/cumulative accuracies use.

**Who labelled the data, and how?**
Annotators were recruited as student research assistants at four institutions in China (three
universities and one research institute). They were trained on an annotation manual and worked
in a purpose-built web platform, annotating the hint text, a free-form guess-card description,
the answer, and a `?` marker when the pun could not be resolved. Items marked `?` went through a
focused adjudication round (several verifiers solved the same batch; answers were cross-checked
against external channels). Three team members operated the platform and ran internal
verification.

**Raw vs released.** All 11,109 annotated rows, including duplicates before grouping, are
retained in the internal corpus archive (`release_all.tsv`); the public release ships
`release_corpus_final.tsv`, the 10,000-item valid set. Summary statistics are in
`stats_summary_final.txt`.

**Inter-annotator agreement.** Annotation is single-pass, so there is no formal two-annotator κ.
The platform does provide a natural repeat sample: 391 puzzles were independently annotated by
two or more different annotators (reposts picked up by different people). Their canonical
answers agree exactly in 342/391 cases (87.5%), and 187 of those pairs also produced identical
guess-card descriptions. Treat this as a lower bound on answer reliability.

## 5. Uses

**What is the dataset designed for?**
Benchmarking vision-language models on two-image Chinese homophone rebuses (single-turn open
decoding and multi-turn interaction with per-position phonological feedback), plus ablations
that separate scene reading, homophone use, world knowledge and trial-and-error.

**Is there anything that should be avoided?**
Do not use the images commercially; do not use the corpus to train systems that generate
deceptive, harassing or culturally abusive content; do not treat the corpus as a representative
sample of Chinese humour (see §2). The non-commercial restriction exists precisely to keep the
third-party images inside the setting their rights holders can be expected to accept.

**Are there tasks for which the dataset should not be used?**
Dialect or Chinese-as-a-second-language evaluation, and any claim about homophone comprehension
in general: the corpus is standard-Mandarin and sample-skewed.

## 6. Distribution

**Under what licence?**
Annotations, metadata, alignment, prompts and code: **CC BY-NC 4.0**.
Images: **© the original WeChat official accounts / illustrators**; we redistribute them for
non-commercial research use only, with attribution.

**How can it be redistributed?**
Keep the attribution file (source account + post URL per item), the licence files, and this
datasheet. Redistribution must stay non-commercial.

**Takedown.** A rights holder can request removal of a single item or of all items from one
account; the affected items are then removed from the release and a new minor version is
published. Contact: see `README.md`.

## 7. Maintenance

- **Versioning:** every file carries a version string; corrections are released as a new minor
  version rather than by editing files in place. See `CHANGELOG.md` (if present) or the release
  notes.
- **Errata:** file an issue or contact the maintainer.
- **Support / contact:** see `README.md`.
- **Canonical documentation:** this file and `README.md`; the paper's Datasheet appendix is a
  summary.
