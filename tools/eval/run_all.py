"""多模型全量驱动：对 N 个模型依次跑完整题库。

前置条件：``<outdir>/items.jsonl`` 与 ``<outdir>/images/`` 已就绪
（由 ``tools/corpus/build_eval10k.py`` 之类生成）。

用法
----
::

    python tools/eval/run_all.py --outdir data/eval10k --models-file models.json

每个模型产出 ``<outdir>/runs/main_<模型标签>.jsonl``，可 ``--resume`` 续跑。

⚠️ 与 ``run_ablation.py`` 同样的教训：本脚本的远端启动等待循环**只在
前台可靠**，脱离终端运行会假死。脱离运行请手动起服务器后直接调
``punquizcn.eval.runner``。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablation import start_model, stop_model
from server_config import (
    ConfigError,
    Endpoint,
    endpoint_from_env,
    load_models,
    remote_from_env,
)


def run_one_model(
    *,
    endpoint: Endpoint,
    tag: str,
    outdir: Path,
    max_turns: int,
    parallel: int,
    feedback: str,
    items_override: Path | None,
    images_override: Path | None,
) -> tuple[bool, float]:
    """跑一个模型的完整题库。返回 (是否成功, 用时秒数)。"""
    rundir = outdir / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    out = rundir / f"main_{tag}.jsonl"
    items = items_override or (outdir / "items.jsonl")
    images = images_override or (outdir / "images")

    cmd = [
        sys.executable,
        "-m",
        "punquizcn.eval.runner",
        "--scheme",
        "A",
        "--items",
        str(items),
        "--images",
        str(images),
        "--max-turns",
        str(max_turns),
        "--max-tokens",
        "32",
        "--parallel",
        str(parallel),
        "--feedback",
        feedback,
        "--base",
        endpoint.base,
        "--out",
        str(out),
        "--resume",
        "--checkpoint-every",
        "50",
    ]
    if endpoint.api_key:
        cmd += ["--api-key", endpoint.api_key]

    print(f"  模型 {tag} -> {out.name}")
    t0 = time.time()
    log = rundir / f"main_{tag}.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
    elapsed = time.time() - t0
    got = 0
    if out.exists():
        with out.open(encoding="utf-8", errors="replace") as fh:
            got = sum(1 for line in fh if line.strip())
    print(f"    -> {got} 条，用时 {elapsed / 3600:.2f} 小时 (exit={proc.returncode})")
    return proc.returncode == 0, elapsed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_all", description="全量评测批量驱动")
    ap.add_argument(
        "--outdir", type=Path, required=True, help="评测数据目录（含 items.jsonl 与 images/）"
    )
    ap.add_argument("--models-file", type=Path, default=None, help="模型清单 JSON")
    ap.add_argument("--models", default=None, help="模型清单内联 JSON（与 --models-file 二选一）")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定模型标签")
    ap.add_argument("--items", type=Path, default=None, help="覆盖 items.jsonl 路径")
    ap.add_argument("--images", type=Path, default=None, help="覆盖 images/ 路径")
    ap.add_argument("--max-turns", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--feedback", default="none")
    ap.add_argument("--dry-run", action="store_true")
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

    if args.only:
        want = set(args.only)
        unknown = want - {m.tag for m in models}
        if unknown:
            print(f"未知模型: {unknown}")
            return 2
        models = [m for m in models if m.tag in want]

    print(f"计划：{len(models)} 个模型")
    print(f"  服务地址: {endpoint.display}")
    print(f"  数据目录: {args.outdir}")
    for spec in models:
        print(f"  {spec.tag}")
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
        try:
            run_one_model(
                endpoint=endpoint,
                tag=spec.tag,
                outdir=args.outdir,
                max_turns=args.max_turns,
                parallel=args.parallel,
                feedback=args.feedback,
                items_override=args.items,
                images_override=args.images,
            )
        except Exception as exc:  # noqa: BLE001 - 单模型失败不该中断整轮
            print(f"  模型 {spec.tag} 异常：{type(exc).__name__}: {exc}")
        stop_model(remote)

    print(f"\n===== 全部结束 {time.strftime('%F %T')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
