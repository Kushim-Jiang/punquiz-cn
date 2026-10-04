"""评测运行器：按方案与档位驱动一次批量评测。

设计要点
--------
* 单条运行是纯函数式的（``Evaluator.run_item``）；批量调度封装在
  ``Experiment`` 类里，便于被其它脚本复用。
* 所有请求走同一个 ``LlamaClient``，它的 base URL 与 API key **必须显式传入**，
  没有内网默认值。
* 结果文件一单位一行（含嵌套 ``turns``），支持 ``--resume`` 断点续跑。

用法
----
::

    # 全量单轮
    python -m punquizcn.eval.runner \\
        --items data/eval10k/items.jsonl \\
        --images data/eval10k/images \\
        --base http://localhost:15021 \\
        --out runs/main.jsonl

    # 分层抽样 500 条/层，用于消融
    python -m punquizcn.eval.runner ... --sample 500 --sample-dim answer_len
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from punquizcn.eval import prompts

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MAX_IMAGE_SIDE = 640
JPEG_QUALITY = 85
TURN_FEEDBACKS = prompts.T1_RETRY_FEEDBACKS

#: 消融档位：档名 -> (CLI 参数, 人类可读说明)
#: 只保留输入档与难度刻度。多轮反馈与提示词排版两组已删除，
#: 理由见 README「设计取舍」一节与论文 §4.4。
ABLATION_TIERS: dict[str, tuple[dict[str, Any], str]] = {
    "inputA": ({"input_mode": "A"}, "两图（真实玩法，参照点）"),
    "inputB": ({"input_mode": "B"}, "两图 + 卡面文字照抄（免 OCR）"),
    "inputC": ({"input_mode": "C"}, "两图 + 人工画面描述（免读图）"),
    "inputD": ({"input_mode": "D"}, "纯文字无图（无视觉下界）"),
    "hintLen": ({"hint_length": True}, "给答案字数"),
    "hintType": ({"hint_type": "成语"}, "给答案类别"),
    "hintBoth": ({"hint_length": True, "hint_type": "成语"}, "字数 + 类别同时给"),
}


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerConfig:
    """远端推理服务配置。base 与 api_key 无默认值，必须显式给出。"""

    base: str
    model: str = "local"
    api_key: str | None = None
    timeout: float = 600.0

    @property
    def chat_url(self) -> str:
        return self.base.rstrip("/") + "/v1/chat/completions"


class LlamaClient:
    """OpenAI 兼容 ``/v1/chat/completions`` 的最小客户端。"""

    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> ServerConfig:
        return self._cfg

    def chat(self, messages: Sequence[dict[str, Any]], max_tokens: int = 256) -> tuple[str, str]:
        """发一次请求，返回 ``(content, finish_reason)``。

        ``finish_reason`` 透传服务端取值（``stop``/``length``/空串），
        用于单独统计截断率。
        """
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = "Bearer " + self._cfg.api_key

        req = urllib.request.Request(
            self._cfg.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._cfg.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return str(content), str(choice.get("finish_reason") or "")


# ---------------------------------------------------------------------------
# 图片缓存
# ---------------------------------------------------------------------------
class ImageCache:
    """惰性编码图片为 data URL，同一路径只编码一次。

    多线程安全：评测是并发的，必须加锁，否则同一张图会被重复编码多次。
    """

    def __init__(self, max_side: int = MAX_IMAGE_SIDE, quality: int = JPEG_QUALITY) -> None:
        self._max_side = max_side
        self._quality = quality
        self._cache: dict[Path, str] = {}
        self._lock = threading.Lock()

    def encode(self, path: str | Path) -> str:
        p = Path(path)
        with self._lock:
            hit = self._cache.get(p)
        if hit is not None:
            return hit
        url = prompts.encode_image_data_url(p, self._max_side, self._quality)
        with self._lock:
            self._cache[p] = url
        return url


# ---------------------------------------------------------------------------
# 题面加载与抽样
# ---------------------------------------------------------------------------
@dataclass
class Item:
    """一道题的评测输入。"""

    id: str
    answer: str
    hint_text: str = ""
    guess_text: str = ""
    split: str = ""
    hint_img: Path | None = None
    guess_img: Path | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any], images_dir: Path | None = None) -> Item:
        iid = str(raw.get("id", ""))
        hint_img = guess_img = None
        if images_dir is not None:
            hint_img = images_dir / f"{iid}__hint.jpg"
            guess_img = images_dir / f"{iid}__guess.jpg"
        return cls(
            id=iid,
            answer=str(raw.get("answer", "")),
            hint_text=str(raw.get("hint_text", "")),
            guess_text=str(raw.get("guess_text", "")),
            split=str(raw.get("split", "")),
            hint_img=hint_img,
            guess_img=guess_img,
        )

    def has_images(self) -> bool:
        return (
            self.hint_img is not None
            and self.guess_img is not None
            and self.hint_img.is_file()
            and self.guess_img.is_file()
        )

    def as_prompt_item(self) -> dict[str, Any]:
        """转成 ``prompts`` 模块期望的字段名。"""
        out: dict[str, Any] = {
            "id": self.id,
            "answer": self.answer,
            "hint_text": self.hint_text,
            "guess_text": self.guess_text,
        }
        if self.hint_img is not None:
            out["hint_img"] = str(self.hint_img)
        if self.guess_img is not None:
            out["guess_img"] = str(self.guess_img)
        return out


def load_items(items_path: Path, images_dir: Path | None, limit: int | None = None) -> list[Item]:
    """读 JSONL 题面；给了 ``images_dir`` 就只保留两图齐备的条目。"""
    items: list[Item] = []
    with items_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(Item.from_json(json.loads(line), images_dir))
    if images_dir is not None:
        before = len(items)
        items = [it for it in items if it.has_images()]
        if len(items) != before:
            print(f"  图片筛选：{before} -> {len(items)} 条（两图齐备）")
    return items[:limit] if limit else items


def _stratum_key(item: Item, dim: str) -> str:
    """分层键。可用维度：``answer_len``（答案字数）、``split``。"""
    if dim == "split":
        return item.split or "__none__"
    n = len(item.answer)
    return f"len{n if n <= 6 else 7}"


def stratified_sample(
    items: Sequence[Item], per_class: int, seed: int, dim: str = "answer_len"
) -> list[Item]:
    """按分层维度抽样，每层至多 ``per_class`` 条。

    固定 seed 保证**每个消融档抽到同一批题**，档间差异才只由所改的
    那一个变量造成。

    .. note::
       语料没有 ``category`` 字段，能用的分层维度只有答案字数与 split。
       ``answer_len`` 实际只有 6 层（2/3/4/5/6/7+ 字），所以每层取 N 条
       得到的总量通常**远小于** 6N——长答案那几层本身题量就少。
    """
    if per_class <= 0:
        return list(items)

    buckets: dict[str, list[Item]] = {}
    for it in items:
        buckets.setdefault(_stratum_key(it, dim), []).append(it)

    rng = random.Random(seed)
    picked: list[Item] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda x: x.id)
        rng.shuffle(pool)
        picked.extend(pool[:per_class])

    order = {it.id: i for i, it in enumerate(items)}
    picked.sort(key=lambda x: order.get(x.id, 0))
    return picked


# ---------------------------------------------------------------------------
# 结果文件读写
# ---------------------------------------------------------------------------
def _unit_finished(record: dict[str, Any], max_turns: int) -> bool:
    """该单位是否已跑完（命中即终止，或已用满轮次）。"""
    if record.get("hit_turn"):
        return True
    return len(record.get("turns") or []) >= max_turns


class ResultStore:
    """结果文件：一单位一行，支持续跑与去重。"""

    def __init__(self, path: Path, max_turns: int) -> None:
        self._path = path
        self._max_turns = max_turns

    @property
    def path(self) -> Path:
        return self._path

    def load_done(self) -> set[tuple[str, str]]:
        """返回已完成的 ``(scheme, id)`` 集合。"""
        if not self._path.is_file():
            return set()
        done: set[tuple[str, str]] = set()
        with self._path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _unit_finished(d, self._max_turns):
                    done.add((str(d.get("scheme", "")), str(d.get("id", ""))))
        return done

    def dedupe(self) -> int:
        """把重复单位合并掉（``--resume`` 会重复追加）。

        保留策略：优先保留「已完成或轮次更多」的那条记录。
        """
        if not self._path.is_file():
            return 0
        best: dict[tuple[str, str], str] = {}
        order: list[tuple[str, str]] = []
        total = 0
        with self._path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(d.get("scheme", "")), str(d.get("id", "")))
                if key not in best:
                    order.append(key)
                    best[key] = line
                else:
                    try:
                        old = json.loads(best[key])
                    except json.JSONDecodeError:
                        best[key] = line
                        continue
                    old_rank = (
                        int(_unit_finished(old, self._max_turns)),
                        len(old.get("turns") or []),
                    )
                    new_rank = (
                        int(_unit_finished(d, self._max_turns)),
                        len(d.get("turns") or []),
                    )
                    if new_rank > old_rank:
                        best[key] = line

        removed = total - len(best)
        if removed <= 0:
            return 0
        tmp = self._path.with_suffix(self._path.suffix + ".dedup")
        with tmp.open("w", encoding="utf-8") as fh:
            for key in order:
                fh.write(best[key] + "\n")
        tmp.replace(self._path)
        return removed

    def append(self, record: dict[str, Any], fh: io.TextIOWrapper) -> None:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# 单条评测
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """一次批量评测的配置。"""

    base: str
    model: str = "local"
    api_key: str | None = None
    max_turns: int = 1
    max_tokens: int = 32
    parallel: int = 4
    input_mode: str = "A"
    output_mode: str = "answer_only"
    hint_length: bool | None = None
    hint_type: str | None = None
    role_mode: str = "system"
    feedback: str = "none"
    check_leak: bool = True

    def ablation_kwargs(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "output_mode": self.output_mode,
            "hint_length": self.hint_length,
            "hint_type": self.hint_type,
            "role_mode": self.role_mode,
        }


class Evaluator:
    """把「一道题 × 一个方案」跑成一条结果记录。"""

    def __init__(self, client: LlamaClient, cfg: RunConfig, images: ImageCache) -> None:
        self._client = client
        self._cfg = cfg
        self._images = images

    def _build_messages(
        self,
        item: Item,
        scheme: str,
        prev: Sequence[str],
        turn: int,
    ) -> list[dict[str, Any]]:
        kw = self._cfg.ablation_kwargs()
        prompt_item = item.as_prompt_item()
        enc = self._images.encode
        if turn == 1:
            return prompts.build_t1(
                prompt_item,
                input_mode=kw["input_mode"],
                output_mode=kw["output_mode"],
                image_encoder=enc,
                hint_length=kw["hint_length"],
                hint_type=kw["hint_type"],
                role_mode=kw["role_mode"],
            )
        return prompts.build_t1_retry(
            prompt_item,
            list(prev),
            feedback=self._cfg.feedback,
            input_mode=kw["input_mode"],
            output_mode=kw["output_mode"],
            image_encoder=enc,
            hint_length=kw["hint_length"],
            hint_type=kw["hint_type"],
            role_mode=kw["role_mode"],
        )

    def run_item(self, item: Item, scheme: str) -> dict[str, Any]:
        prev: list[str] = []
        turns: list[dict[str, Any]] = []
        hit_turn: int | None = None

        for turn in range(1, self._cfg.max_turns + 1):
            prompt_item = item.as_prompt_item()
            messages = self._build_messages(item, scheme, prev, turn)
            if self._cfg.check_leak:
                prompts.assert_no_leak(messages, prompt_item, scheme, turn, prev_answers=prev)
            try:
                raw, finish = self._client.chat(messages, self._cfg.max_tokens)
            except Exception as exc:  # noqa: BLE001 - 单次请求失败记成空回答继续，整个任务不能因此中断
                turns.append(
                    {
                        "turn": turn,
                        "raw": "",
                        "guess": "",
                        "correct": False,
                        "why": "",
                        "finish": "",
                        "error": f"{type(exc).__name__}: {exc}"[:160],
                    }
                )
                break
            guess = prompts.parse_answer_only(raw)
            ok, why = prompts.is_correct(guess, item.answer)
            if ok and hit_turn is None:
                hit_turn = turn
            prev.append(guess or prompts.norm(raw)[:20])
            turns.append(
                {
                    "turn": turn,
                    "raw": raw,
                    "guess": guess,
                    "correct": ok,
                    "why": why,
                    "finish": finish,
                }
            )
            if ok:
                break

        return {
            "scheme": scheme,
            "id": item.id,
            "gold": item.answer,
            "hit_turn": hit_turn,
            "turns": turns,
            "prompt_version": prompts.PROMPT_VERSION,
        }


# ---------------------------------------------------------------------------
# 批量实验
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """一次批量评测：负责调度、续跑、落盘与汇总。"""

    cfg: RunConfig
    items: list[Item]
    out_path: Path
    scheme: str = "A"
    resume: bool = False
    checkpoint_every: int = 50
    _store: ResultStore = field(init=False, repr=False)
    _images: ImageCache = field(init=False, repr=False)
    _evaluator: Evaluator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = ResultStore(self.out_path, self.cfg.max_turns)
        self._images = ImageCache()
        client = LlamaClient(
            ServerConfig(base=self.cfg.base, model=self.cfg.model, api_key=self.cfg.api_key)
        )
        self._evaluator = Evaluator(client, self.cfg, self._images)

    @property
    def store(self) -> ResultStore:
        return self._store

    def plan(self) -> list[Item]:
        """确定本次实际要跑的题（续跑时剔除已完成的）。"""
        if not self.resume:
            return list(self.items)
        done = self._store.load_done()
        todo = [it for it in self.items if (self.scheme, it.id) not in done]
        skipped = len(self.items) - len(todo)
        print(f"  续跑：已完成 {skipped}，剩余 {len(todo)}（共 {len(self.items)}）")
        return todo

    def run(self) -> dict[str, Any]:
        todo = self.plan()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        if not todo:
            print("  没有剩余单位，直接汇总已有结果。")
            return self.summarise()

        removed = self._store.dedupe()
        if removed:
            print(f"  [自愈] 结果文件去重：删掉 {removed} 条重复单位")

        if not self.resume:
            self.out_path.write_text("", encoding="utf-8")

        t0 = time.time()
        n_done = 0
        with cf.ThreadPoolExecutor(max_workers=self.cfg.parallel) as pool:
            futures = {pool.submit(self._evaluator.run_item, it, self.scheme): it for it in todo}
            with self.out_path.open("a", encoding="utf-8") as fh:
                for fut in cf.as_completed(futures):
                    item = futures[fut]
                    try:
                        record = fut.result()
                    except Exception as exc:  # noqa: BLE001 - 单题异常也要落盘成记录，不能弄丢整个批次
                        record = {
                            "scheme": self.scheme,
                            "id": item.id,
                            "gold": item.answer,
                            "hit_turn": None,
                            "turns": [],
                            "error": f"{type(exc).__name__}: {exc}"[:160],
                        }
                    self._store.append(record, fh)
                    n_done += 1
                    if n_done % self.checkpoint_every == 0:
                        print(f"  [{n_done}/{len(todo)}] {int(time.time() - t0)}s")

        dt = time.time() - t0
        print(f"  完成 {n_done} 条，用时 {dt / 60:.1f} 分钟")
        return self.summarise()

    def summarise(self) -> dict[str, Any]:
        """汇总结果文件，返回统计字典。"""
        if not self.out_path.is_file():
            return {}
        first = second = third = 0
        hit = 0
        n = 0
        seen: dict[str, dict[str, Any]] = {}
        with self.out_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen[str(d.get("id", n))] = d

        for d in seen.values():
            n += 1
            turns = d.get("turns") or []
            if len(turns) >= 1 and turns[0].get("correct"):
                first += 1
            if len(turns) >= 2 and turns[1].get("correct"):
                second += 1
            if len(turns) >= 3 and turns[2].get("correct"):
                third += 1
            if d.get("hit_turn"):
                hit += 1

        stats = {
            "n": n,
            "first_turn": first,
            "acc_first": first / n if n else 0.0,
            "second_turn": second,
            "third_turn": third,
            "cumulative_hit": hit,
            "acc_cumulative": hit / n if n else 0.0,
        }
        if n:
            print(
                f"  汇总：n={n}  首轮 {first}（{100 * first / n:.1f}%）  "
                f"累计 {hit}（{100 * hit / n:.1f}%）"
            )
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="punquizcn.eval.runner",
        description="PunQuizCN 评测运行器（单轮主实验与输入档/刻度消融）",
    )
    ap.add_argument("--items", type=Path, required=True, help="题面 JSONL")
    ap.add_argument(
        "--images", type=Path, default=None, help="图片目录（文件名须为 <id>__hint.jpg）"
    )
    ap.add_argument("--out", type=Path, required=True, help="结果 JSONL（一单位一行）")
    ap.add_argument("--base", required=True, help="推理服务地址，如 http://localhost:15021")
    ap.add_argument("--api-key", default=None, help="Bearer token；本地 llama.cpp 通常不需要")
    ap.add_argument("--scheme", default="A", choices=["A"], help="方案（当前仅 A）")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--feedback", default="none", choices=list(TURN_FEEDBACKS))
    # -- 消融旋钮 --
    ap.add_argument("--input-mode", default="A", choices=list(prompts.T1_INPUT_MODES))
    ap.add_argument("--output-mode", default="answer_only", choices=list(prompts.T1_OUTPUT_MODES))
    ap.add_argument("--hint-length", action="store_true", help="给答案字数提示")
    ap.add_argument("--hint-type", default=None, help="给答案类别提示，如 成语")
    ap.add_argument("--role-mode", default="system", choices=list(prompts.T1_ROLE_MODES))
    ap.add_argument(
        "--tier",
        default=None,
        choices=sorted(ABLATION_TIERS),
        help="预设消融档位（会覆盖对应的旋钮参数）",
    )
    # -- 抽样 --
    ap.add_argument("--n", type=int, default=None, help="最多取多少条（默认全量）")
    ap.add_argument("--sample", type=int, default=None, help="分层抽样：每层取多少条（如 500）")
    ap.add_argument("--sample-dim", default="answer_len", choices=["answer_len", "split"])
    ap.add_argument("--sample-seed", type=int, default=20261004)
    # -- 续跑与控制 --
    ap.add_argument("--resume", action="store_true", help="从结果文件断点续跑")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument(
        "--no-leak-check", action="store_true", help="关闭泄漏护栏（仅用于验证护栏本身）"
    )
    return ap


def config_from_args(args: argparse.Namespace) -> RunConfig:
    knobs: dict[str, Any] = {
        "input_mode": args.input_mode,
        "hint_length": True if args.hint_length else None,
        "hint_type": args.hint_type,
    }
    if args.tier:
        knobs.update(ABLATION_TIERS[args.tier][0])
    return RunConfig(
        base=args.base,
        model=args.model,
        api_key=args.api_key,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        input_mode=knobs["input_mode"],
        output_mode=args.output_mode,
        hint_length=knobs.get("hint_length"),
        hint_type=knobs.get("hint_type"),
        role_mode=args.role_mode,
        feedback=args.feedback,
        check_leak=not args.no_leak_check,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    items = load_items(args.items, args.images, args.n)
    if args.sample:
        before = len(items)
        items = stratified_sample(items, args.sample, args.sample_seed, args.sample_dim)
        print(f"  分层抽样（{args.sample_dim}）：{before} -> {len(items)} 条")
    if not items:
        print("没有可评测的题（检查 --items / --images）")
        return 2

    tier = f" 档={args.tier}" if args.tier else ""
    print(
        f"题 {len(items)} 条 | 方案 {args.scheme} | {cfg.max_turns} 轮 | 并发 {cfg.parallel}"
        f"{tier} | 输入档 {cfg.input_mode}"
    )

    exp = Experiment(
        cfg=cfg,
        items=items,
        out_path=args.out,
        scheme=args.scheme,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )
    exp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
