# PunQuizCN

Evaluation code for **PunQuizCN**, a large-scale Chinese multimodal
homophone-rebus corpus.

> **Data is not in this repository.** The corpus is too large to ship in git;
> it is hosted separately. See [Data](#data) below. This repo contains the
> **code**: prompt construction, scoring, leak guards, the experiment runner,
> and the corpus-building / literature tooling.

---

## What this is

The task is the "two-image homophone rebus" (谐音梗) format that is ubiquitous
on Chinese social media. A puzzle presents two image panels: the first is
labelled with the name of an object (`这是X`), the second must be guessed
(`这是___`). Reading the two names aloud and concatenating them yields a
homophone of a common word, idiom, person or place name — the answer.

Because the answer is defined **by sound**, not by characters, scoring is
three-tiered rather than exact-match:

| tier | meaning | example (gold `葫芦娃`) |
|---|---|---|
| `exact` | same characters | `葫芦娃` |
| `homophone` | every syllable identical incl. tone, different characters | `圆周律` vs gold `圆周率` |
| `near` | same initials/finals but ≥1 tone differs | `狐鹿袜` vs gold `葫芦娃` (`wa4` vs `wa2`) |
| `no` | otherwise | `完全无关` |

Only `exact` + `homophone` count toward the headline accuracy; `near` is
reported separately because accepting it would be too lenient.

## Install

Requires **Python ≥ 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kushim-Jiang/punquiz-cn.git
cd punquiz-cn
uv sync
```

Runtime dependencies are deliberately minimal — only `pypinyin` (phonology)
and `pillow` (image downscaling). Everything else is stdlib.

## Quick start

```bash
# Run the self-tests (no server needed)
uv run pytest

# Print the prompt suite and run its built-in self-checks
uv run python -m punquizcn.eval.prompts
```

## Layout

```
src/punquizcn/eval/
    prompts.py    Prompt suite, normalization, 3-tier scoring, leak guard
    runner.py     Batch runner: client, image cache, resume, sampling
tools/
    eval/         Remote-server drivers (ablation, full run)
    corpus/       Corpus build + annotation-DB access
tests/
    test_smoke.py End-to-end checks over the 13 open samples
data/sample/      13 public sample items + 26 images
```

## Using the library

```python
from punquizcn.eval import prompts

# Build first-turn messages for one item
item = {
    "id": "42",
    "hint_img": "data/sample/images/159__hint.jpg",
    "guess_img": "data/sample/images/159__guess.jpg",
    "hint_text": "这是π",
    "guess_text": "翻转了180度的π",
    "answer": "反派",
}
messages = prompts.build_t1(item, input_mode="A", output_mode="answer_only")

# The leak guard is a hard assertion, not a warning: it raises if the answer,
# its syllables, or its split initials/finals/tones appear in the prompt.
prompts.assert_no_leak(messages, item, scheme="A", turn=1, prev_answers=[])

# Score a model answer
ok, why = prompts.is_correct("狐鹿袜", "葫芦娃")  # (True, 'near')
```

### Leak guard

This is the part of the codebase that took the most iteration, and it is worth
understanding before you rely on it.

An LLM benchmark whose prompt contains the answer is worthless, so the guard is
an `assert`, not a log line. It checks three things:

1. The answer's characters do not appear verbatim.
2. The answer's per-syllable toned form (`shan1`) does not appear.
3. The initials/finals/tone written **separately** do not appear
   (e.g. "声母 sh 韵母 an 声调 1").

Two false-positive classes had to be handled, both discovered the hard way:

* **The model's own prior guess.** In multi-turn mode the transcript contains
  what the model guessed last turn — which may be the answer. That is legal;
  `prev_answers` is stripped before checking.
* **Template boilerplate.** The template says "不要用 ``` 代码块包裹"
  ("don't wrap in a code block"), which contains the answer `代码`. The prompt
  is meant to be shown to the model, so this is not a leak. All module-level
  template constants are stripped first.

See `assert_no_leak` and `strip_prompt_boilerplate` in `prompts.py`; the
regression test for the `代码` case is
`tests/test_smoke.py::test_leak_guard_catches_template_phrase`.

## Running experiments against a GPU server

The drivers in `tools/eval/` expect a remote box running
[llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`, reached
over SSH. **All hostnames, keys and API tokens come from the environment —
nothing is hard-coded.**

```bash
export PUNQUIZCN_HOST=user@gpu-box.example.com
export PUNQUIZCN_SSH_KEY=~/.ssh/id_ed25519      # optional if you have an agent
export PUNQUIZCN_BASE=http://gpu-box.example.com:15021
export PUNQUIZCN_API_KEY=...                    # optional; llama.cpp needs none
```

Model `.gguf` paths differ per machine, so they are declared in a JSON file
rather than baked in. See `tools/eval/models.example.json`:

```json
[
  {
    "tag": "qwen3vl_8b",
    "args": "-m $HOME/models/qwen3vl8b.gguf --mmproj $HOME/models/qwen3vl8b_mmproj.gguf --ctx-size 32768 --parallel 4"
  }
]
```

Then:

```bash
# Inspect the plan without running anything
uv run python tools/eval/run_ablation.py \
    --outdir data/eval10k --models-file models.json --dry-run

# 7-tier ablation (4 input tiers + 3 difficulty dials)
uv run python tools/eval/run_ablation.py \
    --outdir data/eval10k --models-file models.json
```

Output goes to `<outdir>/ablation/abl_<tier>_<model>.jsonl`, one unit per line.
Every run supports `--resume`, so an interrupted job continues where it stopped.

### Design note: why only 7 ablation tiers

Two groups were designed and then **deliberately dropped**:

* **Feedback granularity** (`none`/`bin`/`tri`/`phon`/`phon_char`/`phon_near`).
  Measured feedback gain was only **+0.35 to +2.25 pp**, against first-turn
  accuracies of 0.4–9.6%, with sharply diminishing returns on turns 2 and 3.
  Spending 6 tiers × 6 models would only re-derive "feedback helps a little" —
  which the existing ~60k-item run already establishes. The result is reported
  in the paper as an informative **negative finding**, not as untested.
* **Prompt-layout variants** (schemes A–E). These contribute one sentence
  ("results don't depend on one particular phrasing") and their letters
  collided confusingly with the input tiers A–D.

The retained tiers are 4 input modes (`A` two images / `B` two images + panel
text / `C` two images + scene description / `D` text only, no images) and 3
difficulty dials (word-count hint, category hint, both).

### Operational warning

The drivers start `llama-server` over SSH and then poll for readiness. **That
wait loop only works in the foreground.** Run detached (e.g. via Windows Task
Scheduler) and the server comes up but no requests are ever sent — the GPU sits
at 0%. If you need to detach, start the server yourself and invoke
`python -m punquizcn.eval.runner` directly.

Two related failure modes are worth knowing about: `llama-server` routinely
becomes orphaned (client dies, server keeps holding ~12 GB VRAM at 0% GPU — the
tell is zero established connections on the port **and** 0% utilisation, in
which case kill it), and stratified sampling over `answer_len` yields far fewer
items than `per_class × 6` because the 5/6/7+-character buckets are genuinely
small (102/16/45 items).

## Data

The corpus is **not** distributed in this repository. Statistics for the
released set (`release_corpus_final.tsv`, 10,000 valid items):

* 4,042 two-character / 3,107 three-character / 2,700 four-character answers
  (plus a long tail of 5–26 character multi-word answers)
* 4,387 unique answers
* split: train 8,026 / dev 979 / test 995
* 197 items containing Latin letters, 312 containing digits
* 9,192 items whose hint panel begins with `这是` ("this is")

Point the code at your copy with either an environment variable or a path:

```bash
export PUNQUIZCN_CORPUS=/path/to/release_corpus_final.tsv
```

`prompts.corpus_path()` resolves `$PUNQUIZCN_CORPUS` first, then falls back to
`<repo>/data/release_corpus_final.tsv`. The corpus is only needed by the
`_demo()` self-check that verifies the prompt's example words are not real
answers; every other code path works without it.

`data/sample/` ships 13 real items with their 26 images so the test suite and
the API examples have something concrete to run on. Each line also carries a
`url` field: the original WeChat article the item was collected from, so
provenance is traceable back to the source post.

Note on `answer` formatting: the sample's `id` 4945 is stored as
`栓Q（thank you）`, because that is verbatim what the corpus contains — the
parenthetical is the corpus's own gloss of the pun, not part of the answer.
Scoring handles it: `is_correct("栓Q", "栓Q（thank you）")` returns
`(True, 'homophone')`, so a model that emits the clean answer is still counted
correct. Only the tier label differs from `exact`.

### Building the corpus

`tools/corpus/` contains the pipeline. It talks to the annotation database
stored on a separate host, so those connections are configured by environment
variable too — **no credentials are in this repository**:

```bash
export PUNQUIZCN_SRC_HOST=anno.example.com
export PUNQUIZCN_SRC_USER=alice
export PUNQUIZCN_SRC_KEY=~/.ssh/id_ed25519   # preferred over a password
# export PUNQUIZCN_SRC_PASSWORD=...         # alternative, less safe
export PUNQUIZCN_SRC_BASE=/srv/annoSys

uv run python tools/corpus/inspect_db.py            # explore the annotation DB
uv run python tools/corpus/sftp_download.py --local-root /path/to/workspace
uv run python tools/corpus/prepare_benchmark.py --data-dir data --split test
```

`--local-root` is required rather than defaulted, so no personal path is
baked into the code.

## Data inventory

### Item identifiers

All 10,000 valid items are numbered `q00001`-`q10000`. Numbers are assigned
by data split (train then dev then test) and, within a split, by ascending
original record id, so the identifier itself encodes the split and the
assignment is fully reproducible:

| split | identifier range | items |
|---|---|---|
| train | `q00001`-`q08026` | 8,026 |
| dev | `q08027`-`q09005` | 979 |
| test | `q09006`-`q10000` | 995 |

The bidirectional map between identifiers and original ids ships as
`qid_map.tsv` (original id, identifier, split, answer, answer category,
answer length, both descriptions). **Every row of every released artefact
carries both the identifier and the original id**, so any reported result can
be traced to a specific item and any item can be looked up across all
experiments.

### Released artefacts

All experiment artefacts are released with the resource:

| artefact | contents |
|---|---|
| `qid_map.tsv` | identifier map (10,000 rows) |
| `items_qid.jsonl` | item statements with identifiers, ready to load for evaluation |
| `main_results.jsonl` | full per-item main results (60,000 rows = 6 models x 10,000 items); each row gives identifier, model, judging outcome, and the parsed guess per turn |
| `ablation_all.jsonl` | full per-item ablation results (69,348 rows = 7 conditions x 6 models); same fields plus the condition name |
| `tables/` | every statistics table in the paper, in both LaTeX and TSV |

### Reproducibility

All experiments run locally on our own hardware under a single `llama.cpp`
build, greedily decoded (T=0) with reasoning disabled. Every item is
evaluated under the same prompt template, inference engine, and sampling
parameters; the only thing that varies between runs is which weights are
loaded. Both images are re-sent each turn, and no hidden state is carried
across turns.

**Sampling caveat.** The ablations use a fixed-seed stratified sample by
answer length (`--sample 500`), about 1,650 items per condition. Because the
sampling depends on item ordering in the list, *any* addition or removal
shifts the sample wholesale - in auditing, adding or removing just 10 items
changed about 35% of the sampled entries. All ablation results therefore come
from a single pass over one corpus snapshot and are never mixed across
versions.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src              # strict type check
uv run pytest                # tests
```

Style conventions enforced by the lint config:

* **`pathlib` everywhere** — `os.path` is a lint error (`PTH`).
* **Full type annotations** on every function (`ANN`), except in `tools/` and
  `tests/` where scripts are allowed to omit them.
* **Modern generics** — builtin `dict`/`list`/`tuple`, and `X | None` rather
  than `Optional[X]`.
* `E501` (line length) is **off**: the prompt templates and the Chinese
  comments are line-sensitive, and wrapping them hurts more than it helps.
* The full-width punctuation rules `RUF001/002/003` are off for the same
  reason — this codebase is legitimately full of Chinese text.

`mypy` runs in `strict` mode over `src/` only.

## Citing

```bibtex
@inproceedings{punquizcn,
  title     = {PunQuizCN: A Benchmark for Vision-Language Models on Chinese Two-Image Homophone Rebuses},
  author    = {Jiang, Kushim and others},
  booktitle = {To appear},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
