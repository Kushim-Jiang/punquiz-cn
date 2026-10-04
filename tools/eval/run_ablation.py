"""消融实验批量驱动：7 档 x N 模型。

设计（经作者确认，删除了两组消融）
----------------------------------
被删除的两组，理由都写在这里，免得日后有人重新加回来：

* **反馈粒度**（none/bin/tri/phon/phon_char/phon_near）
  实测反馈增益只有 +0.35~+2.25 pp，而首轮绝对准确率本身只有 0.4%~9.6%；
  第 2、3 轮的边际收益还急剧衰减。跑满 6 档 x 6 模型也只能得出「反馈帮助
  很小」这个结论——而这一点用已有的 6 万条数据就能证明。该结论作为
  **负面发现**写进论文，不当成未测项。
* **排版档**（scheme A~E）
  对论文只贡献「结果不依赖某一种写法」一句话，而且它的字母 A~E 与输入档
  A~D 撞车，很容易混淆。

保留的两组
----------
* 消融一·输入档：``A``（两图）/ ``B``（两图+卡面文字）/ ``C``（两图+画面描述）
  / ``D``（纯文字无图）
* 消融二·难度刻度：给字数 / 给类别 / 两者同时给

用法
----
::

    # 先看计划（不跑）
    python tools/eval/run_ablation.py --dry-run --models-file models.json

    # 只跑指定档
    python tools/eval/run_ablation.py --only inputB inputC --models-file models.json

    # 全跑
    python tools/eval/run_ablation.py --models-file models.json

每个档产出 ``<outdir>/abl_<档名>_<模型>.jsonl``；文件名带模型标签，
所以可以逐模型累积并 ``--resume`` 续跑。

⚠️ 关于脱离终端运行
-------------------
本脚本用 ssh + 远端等待循环启动 llama-server，**只在前台可靠**。
用计划任务（schtasks）之类脱离终端的方式跑时，等待循环会假死——服务器
起来了，但没有任何请求发出去（GPU 占用 0%）。脱离运行时请手动起服务器，
然后直接调 ``punquizcn.eval.runner``。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server_config import (
    ConfigError,
    Endpoint,
    ModelSpec,
    Remote,
    endpoint_from_env,
    load_models,
    remote_from_env,
)

# ---------------------------------------------------------------------------
# 档位定义：(档名, CLI 参数, 说明)
# ---------------------------------------------------------------------------
ABLATIONS: list[tuple[str, list[str], str]] = [
    # ---- 消融一：输入档 ----
    ("inputA", ["--input-mode", "A"], "输入档 A：两图（真实玩法，参照点）"),
    ("inputB", ["--input-mode", "B"], "输入档 B：两图 + 卡面文字照抄（免 OCR）"),
    ("inputC", ["--input-mode", "C"], "输入档 C：两图 + 人工画面描述（免读图）"),
    ("inputD", ["--input-mode", "D"], "输入档 D：纯文字无图（无视觉下界）"),
    # ---- 消融二：难度刻度 ----
    ("hintLen", ["--hint-length", "on"], "刻度：给答案字数"),
    ("hintType", ["--hint-type", "成语"], "刻度：给答案类别"),
    ("hintBoth", ["--hint-length", "on", "--hint-type", "成语"], "刻度：字数 + 类别同时给"),
]


def sh(remote: Remote, cmd: str, timeout: int = 900) -> str:
    """在远端跑一条命令，返回 stdout+stderr。"""
    proc = subprocess.run(
        [*remote.ssh_base(), cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def start_model(remote: Remote, spec: ModelSpec) -> bool:
    """起远端 llama-server 并等它进入就绪状态。

    ⚠️ 这里的 ssh + bash 等待循环**只在前台可靠**；脱离终端运行会假死。
    详见模块 docstring 的「关于脱离终端运行」。
    """
    script = f"""#!/bin/bash
BIN={remote.llama_server}
pkill -f llama-server 2>/dev/null; sleep 4
nohup $BIN {spec.args} -ngl 999 --batch-size 2048 --ubatch-size 2048 \\
  --flash-attn on --host 0.0.0.0 --port {remote.port} --reasoning off --no-warmup \\
  > $HOME/prod_{spec.tag}.log 2>&1 &
for i in $(seq 1 90); do
  sleep 5
  grep -q "listening on http" $HOME/prod_{spec.tag}.log 2>/dev/null && {{ echo READY; exit 0; }}
  pgrep -f llama-server >/dev/null || {{ echo DEAD; tail -6 $HOME/prod_{spec.tag}.log; exit 1; }}
done
echo TIMEOUT; exit 1
"""
    out = sh(remote, script, timeout=900)
    ok = "READY" in out
    status = "OK" if ok else out.strip()[-300:]
    print(f"  启动服务器 {spec.tag}: {status}")
    return ok


def stop_model(remote: Remote) -> None:
    """杀掉远端所有 llama-server。"""
    sh(remote, "pkill -f llama-server 2>/dev/null; sleep 3; echo STOPPED", timeout=120)


def run_one_ablation(
    *,
    endpoint: Endpoint,
    ab_name: str,
    extra: list[str],
    model_tag: str,
    n: int,
    sample: int,
    items: Path,
    images: Path,
    outdir: Path,
) -> tuple[bool, float]:
    """跑一个档 x 一个模型。返回 (是否成功, 用时秒数)。"""
    abldir = outdir / "ablation"
    abldir.mkdir(parents=True, exist_ok=True)
    out = abldir / f"abl_{ab_name}_{model_tag}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "punquizcn.eval.runner",
        "--scheme",
        "A",
        "--n",
        str(n),
        "--max-turns",
        "1",  # 消融全部单轮（不再涉及反馈）
        "--max-tokens",
        "32",
        "--parallel",
        "4",
        "--base",
        endpoint.base,
        "--items",
        str(items),
        "--images",
        str(images),
        "--sample",
        str(sample),
        "--sample-dim",
        "answer_len",
        "--resume",
        "--checkpoint-every",
        "50",
        "--out",
        str(out),
        *extra,
    ]
    if endpoint.api_key:
        cmd += ["--api-key", endpoint.api_key]
    print(f"  运行档 {ab_name} / 模型 {model_tag} -> {out.name}")
    t0 = time.time()
    log = abldir / f"abl_{ab_name}_{model_tag}.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
    elapsed = time.time() - t0
    got = 0
    if out.exists():
        with out.open(encoding="utf-8", errors="replace") as fh:
            got = sum(1 for line in fh if line.strip())
    print(f"    -> {got} 条，用时 {elapsed / 60:.1f} 分钟 (exit={proc.returncode})")
    return proc.returncode == 0, elapsed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_ablation",
        description="消融实验批量驱动（7 档: 4 输入档 + 3 难度刻度）",
    )
    ap.add_argument(
        "--outdir", type=Path, required=True, help="评测数据目录（含 items.jsonl 与 images/）"
    )
    ap.add_argument("--models-file", type=Path, default=None, help="模型清单 JSON")
    ap.add_argument("--models", default=None, help="模型清单内联 JSON（与 --models-file 二选一）")
    ap.add_argument("--n", type=int, default=10000, help="交给 runner 的题量上限")
    ap.add_argument(
        "--sample", type=int, default=500, help="每层抽多少条（answer_len 分层，共 6 层）"
    )
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定档名（如 inputB inputC）")
    ap.add_argument("--models-only", nargs="*", default=None, help="只跑指定模型标签")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    try:
        remote = remote_from_env()
        endpoint = endpoint_from_env()
        models = load_models(args.models_file, args.models)
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 2

    todo = ABLATIONS
    if args.only:
        want = set(args.only)
        unknown = want - {x[0] for x in ABLATIONS}
        if unknown:
            print(f"未知档名: {unknown}")
            return 2
        todo = [x for x in ABLATIONS if x[0] in want]

    if args.models_only:
        want = set(args.models_only)
        unknown = want - {m.tag for m in models}
        if unknown:
            print(f"未知模型: {unknown}")
            return 2
        models = [m for m in models if m.tag in want]

    items = args.outdir / "items.jsonl"
    images = args.outdir / "images"

    total = len(todo) * len(models)
    print(f"消融计划：{len(todo)} 档 x {len(models)} 模型 = {total} 次运行")
    print(f"  服务地址: {endpoint.display}")
    print(f"  数据目录: {args.outdir}")
    # ⚠️ 实际条数 < 每层 x 层数：answer_len 名义上 6 层，但 5/6/7+ 三层的题量
    #    只有 102/16/45，取不满每层配额。每层 500 时实测约 1650 条/档。
    print(
        f"  每层 {args.sample} 条（answer_len 分层；因长答案层题量不足，"
        f"实测约 1650 条/档而非 {args.sample * 6}）"
    )
    print()
    for name, extra, desc in todo:
        print(f"  {name:<10} {desc:<46} {' '.join(extra)}")
    print()
    for spec in models:
        print(f"  模型 {spec.tag}")
    if args.dry_run:
        return 0

    print("\n" + "=" * 74)
    print(f"开始，时间 {time.strftime('%F %T')}")
    print("=" * 74)

    for spec in models:
        print("\n" + "=" * 74)
        print(f"模型 {spec.tag}  {time.strftime('%F %T')}")
        print("=" * 74)
        if not start_model(remote, spec):
            print("  服务器启动失败，跳过该模型")
            continue
        for name, extra, _ in todo:
            try:
                run_one_ablation(
                    endpoint=endpoint,
                    ab_name=name,
                    extra=extra,
                    model_tag=spec.tag,
                    n=args.n,
                    sample=args.sample,
                    items=items,
                    images=images,
                    outdir=args.outdir,
                )
            except Exception as exc:  # noqa: BLE001 - 单档失败不该中断整轮
                print(f"  档 {name} 异常，继续下一档：{type(exc).__name__}: {exc}")
        stop_model(remote)

    print(f"\n===== 全部结束 {time.strftime('%F %T')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
