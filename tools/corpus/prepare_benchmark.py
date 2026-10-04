"""从发布语料生成可复现的评测输入。

产出（默认 ``test`` 分片）
--------------------------
``benchmark_t1.jsonl``
    双图开放解码（T1 / 输入档 A）主任务。
``benchmark_t4.jsonl``
    逐位「选字填空」（音→字判别）。
``benchmark_t1_ablation.jsonl``
    三种输入设置的消融条目（纯文本形态，供无图 baseline 用）。

前置条件
--------
1. 已跑过 ``export_release.py --decode``，面板图落在 ``<data>/release/images/``。
   没解码也能出文件，但图片路径会指向不存在的文件。
2. 装了 ``pypinyin``（构造 T4 的同音干扰项要用）。

用法
----
::

    python tools/corpus/prepare_benchmark.py --data-dir data --split test
    python tools/corpus/prepare_benchmark.py --data-dir data --split test --limit 100

术语对应（论文 → 文件）：提示卡 S → ``{id}__hint.jpg``，猜测卡 Q →
``{id}__guess.jpg``，S'=``hint_text``，Q'=``guess_text``，A=``answer``。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r"[\u4e00-\u9fff]")

#: T1 三种输入设置的固定指令模板（论文里作为「输入档」消融）。
SETTING_IMAGE_ONLY = (
    "看图猜谐音梗：下面两张图是一道谜题。图1卡上通常写着“这是X”（X是图里物体的名字），"
    "图2卡要你猜“这是___”。请结合两图，猜出隐藏的谐音答案（一个词/成语/人名等）。"
    "只输出答案本身。"
)
SETTING_VISUAL_HINT = (
    "看图猜谐音梗：图1卡是“{hint}”，图2卡请观察画面猜出。"
    "把“图1的名字”与“图2的名字”读出来拼在一起，就是谐音答案。只输出答案本身。"
)
SETTING_PLUS_DESC = (
    "看图猜谐音梗：图1说明：{hint}；图2画面描述：{guess}。"
    "请结合图片与说明猜出谐音答案。只输出答案本身。"
)
T4_INSTRUCTION = (
    "下面是一道谐音谜题的两张图。已按顺序给出每位的候选字（每行一位置），"
    "请为答案的每一“位”选出一个字，按顺序输出。"
)

T4_DISTRACTORS_MAX = 7


def toneless_syllable(ch: str) -> str | None:
    """取单字的第一读音（无调）。取不到返回 None。"""
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return None
    try:
        result = pinyin(ch, style=Style.NORMAL, heteronym=True)
        if result and result[0]:
            return str(result[0][0]).strip().lower()
    except Exception:  # noqa: BLE001 - pypinyin 对个别生僻字会抛，按「取不到」处理
        return None
    return None


class Corpus:
    """发布语料的一层薄封装：按列名取值，避免到处写 ``idx["..."]``。"""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    @classmethod
    def load(cls, path: Path) -> Corpus:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if not rows:
            raise ValueError(f"语料为空：{path}")
        return cls(rows)

    def non_placeholder(self) -> list[dict[str, str]]:
        """剔掉占位行。"""
        return [r for r in self.rows if r.get("is_placeholder") == "0"]

    def for_split(self, split: str) -> list[dict[str, str]]:
        return [r for r in self.non_placeholder() if r.get("split") == split]

    def syllable_family(self) -> dict[str, set[str]]:
        """语料内 音节 → 该读音下出现过的字。用来构造同音干扰项。"""
        family: dict[str, set[str]] = defaultdict(set)
        for row in self.non_placeholder():
            for ch in CJK.findall(row.get("answer", "")):
                syl = toneless_syllable(ch)
                if syl:
                    family[syl].add(ch)
        return family


def build_t1_rows(data: list[dict[str, str]], image_dir: Path) -> list[dict[str, Any]]:
    """构造 T1 主任务条目。"""
    out: list[dict[str, Any]] = []
    for row in data:
        iid = row["id"]
        out.append(
            {
                "id": iid,
                "account": row.get("account", ""),
                "hint_img": str(image_dir / f"{iid}__hint.jpg"),
                "guess_img": str(image_dir / f"{iid}__guess.jpg"),
                "hint_text": row.get("hint_text", ""),
                "guess_text": row.get("guess_text", ""),
                "answer": row.get("answer", ""),
                # 三种输入设置（论文消融用）
                "setting_visual": SETTING_IMAGE_ONLY,
                "setting_visual_hint": SETTING_VISUAL_HINT,
                "setting_plus_desc": SETTING_PLUS_DESC,
            }
        )
    return out


def build_t4_rows(
    data: list[dict[str, str]],
    family: dict[str, set[str]],
    image_dir: Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """构造 T4 逐位选字条目（每位：gold + 同音/随机干扰项）。"""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for row in data:
        answer = row.get("answer", "")
        chars = CJK.findall(answer)
        if not chars:
            continue
        positions: list[dict[str, Any]] = []
        for ch in chars:
            syl = toneless_syllable(ch)
            pool = sorted((family.get(syl or "") or set()) - {ch})
            distractors = rng.sample(pool, min(T4_DISTRACTORS_MAX, len(pool))) if pool else []
            positions.append({"gold": ch, "syllable": syl or "", "candidates": [ch, *distractors]})
        iid = row["id"]
        out.append(
            {
                "id": iid,
                "hint_img": str(image_dir / f"{iid}__hint.jpg"),
                "guess_img": str(image_dir / f"{iid}__guess.jpg"),
                "hint_text": row.get("hint_text", ""),
                "guess_text": row.get("guess_text", ""),
                "answer": answer,
                "positions": positions,
                "instruction": T4_INSTRUCTION,
            }
        )
    return out


def build_ablation_rows(t1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 T1 条目转成三种设置的纯文本形态（供无图 baseline）。"""
    out: list[dict[str, Any]] = []
    for entry in t1_rows:
        hint = entry["hint_text"]
        guess = entry["guess_text"]
        out.append(
            {
                "id": entry["id"],
                "answer": entry["answer"],
                "A_image_only": entry["setting_visual"],
                "B_label": entry["setting_visual_hint"].format(hint=hint),
                "C_text_only": entry["setting_plus_desc"].format(hint=hint, guess=guess),
                "texts": {"hint": hint, "guess": guess},
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """一单位一行写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="prepare_benchmark", description="从 release_corpus.tsv 生成评测输入"
    )
    ap.add_argument("--data-dir", type=Path, required=True, help="data/ 目录")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 条（调试用）")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    release = args.data_dir / "release"
    corpus_path = release / "release_corpus.tsv"
    image_dir = release / "images"
    if not corpus_path.is_file():
        print(f"语料不在：{corpus_path}")
        return 1

    corpus = Corpus.load(corpus_path)
    data = corpus.for_split(args.split)
    if args.limit:
        data = data[: args.limit]
    print(f"条目：{len(data)}（split={args.split}）")

    family = corpus.syllable_family()

    t1_rows = build_t1_rows(data, image_dir)
    write_jsonl(release / "benchmark_t1.jsonl", t1_rows)
    print(f"写出 benchmark_t1.jsonl（{len(t1_rows)}）")

    t4_rows = build_t4_rows(data, family, image_dir)
    write_jsonl(release / "benchmark_t4.jsonl", t4_rows)
    print(f"写出 benchmark_t4.jsonl（{len(t4_rows)}）")

    abl_rows = build_ablation_rows(t1_rows)
    write_jsonl(release / "benchmark_t1_ablation.jsonl", abl_rows)
    print(f"写出 benchmark_t1_ablation.jsonl（{len(abl_rows)}）")

    print(
        "提示：T1 评测还需要一个「等价答案判定器」（同义/近音可接受集合），"
        "本仓库提供 punquizcn.eval.prompts.is_correct 作为三级判分。"
    )
    print(
        f"图片路径指向 {image_dir}/{{id}}__{{hint,guess}}.jpg —— 若未解码，先跑 export_release.py --decode"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
