"""把远端标注数据（data / data_ok 两个目录）用 SFTP 拉到本地。

特性
----
* 多线程下载（多个并行 SFTP 连接）
* 断点续传：本地已存在且大小一致的文件跳过；大小不符的从断点续传
* 瞬时错误自动重试
* 进度输出

远端根目录与凭据全部来自环境变量（见 ``source_config.py``），
本地落盘位置由 ``--local-root`` 指定。

用法
----
::

    python tools/corpus/sftp_download.py --local-root D:/Github/pun
"""

from __future__ import annotations

import argparse
import queue
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from source_config import ConfigError, SourceConfig, connect, source_from_env

FOLDERS = ["data", "data_ok"]
WORKERS = 6
RETRIES = 4
CHUNK = 1024 * 1024


def build_file_list(sftp: Any, remote_dir: str, local_dir: Path) -> list[tuple[str, Path, int]]:
    """递归列出远端文件，返回 ``[(远端路径, 本地路径, 字节数), ...]``。"""
    files: list[tuple[str, Path, int]] = []
    stack = [remote_dir]
    while stack:
        cur = stack.pop()
        rel = cur[len(remote_dir) :].lstrip("/")
        local_cur = local_dir / rel if rel else local_dir
        try:
            entries = sftp.listdir_attr(cur)
        except Exception as exc:  # noqa: BLE001 - 单个目录列不出来不该放弃整批
            print(f"[WARN] 无法列出 {cur}: {exc}", file=sys.stderr)
            continue
        for attr in entries:
            full = f"{cur}/{attr.filename}"
            if stat.S_ISDIR(attr.st_mode):
                stack.append(full)
            elif stat.S_ISREG(attr.st_mode):
                files.append((full, local_cur / attr.filename, int(attr.st_size)))
    return files


def download_file(
    sftp: Any,
    remote_path: str,
    local_path: Path,
    expected_size: int,
    index: int,
    total: int,
    lock: threading.Lock,
    done_counter: list[int],
) -> bool:
    """下载单个文件，带断点续传与重试。成功返回 True。"""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size == expected_size:
        with lock:
            done_counter[0] += 1
            if index % 200 == 0:
                print(f"[{done_counter[0]}/{total}] 跳过（已存在）{local_path.name}")
        return True

    for attempt in range(1, RETRIES + 1):
        try:
            offset = 0
            mode = "wb"
            if local_path.exists():
                offset = local_path.stat().st_size
                if offset > expected_size:
                    local_path.unlink()
                    offset = 0
                elif offset > 0:
                    mode = "ab"

            with sftp.open(remote_path, "rb") as remote, local_path.open(mode) as local:
                if offset > 0:
                    remote.prefetch()
                    remote.seek(offset)
                remaining = expected_size - offset
                while remaining > 0:
                    chunk = remote.read(CHUNK)
                    if not chunk:
                        break
                    local.write(chunk)
                    remaining -= len(chunk)

            if local_path.stat().st_size == expected_size:
                with lock:
                    done_counter[0] += 1
                    if index % 200 == 0:
                        print(f"[{done_counter[0]}/{total}] OK {local_path.name}")
                return True
            local_path.unlink()  # 大小不符 -> 下一轮从头来
        except Exception as exc:  # noqa: BLE001 - 网络抖动要重试，重试耗尽才判失败
            if attempt < RETRIES:
                time.sleep(1.5 * attempt)
            else:
                with lock:
                    done_counter[0] += 1
                    print(f"[ERROR] {remote_path}: {exc}", file=sys.stderr)
    return False


def worker(
    cfg: SourceConfig,
    work_queue: queue.Queue[tuple[str, Path, int, int]],
    total: int,
    lock: threading.Lock,
    done_counter: list[int],
    failed: list[str],
) -> None:
    """一个下载线程：自己开一条 SFTP 连接。"""
    ssh = connect(cfg)
    try:
        sftp = ssh.open_sftp()
        while True:
            try:
                remote_path, local_path, size, index = work_queue.get_nowait()
            except queue.Empty:
                break
            if not download_file(
                sftp, remote_path, local_path, size, index, total, lock, done_counter
            ):
                failed.append(remote_path)
            work_queue.task_done()
        sftp.close()
    finally:
        ssh.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sftp_download", description="从标注服务器拉取 data/ 与 data_ok/"
    )
    ap.add_argument("--local-root", type=Path, required=True, help="本地落盘的仓库根目录")
    ap.add_argument("--workers", type=int, default=WORKERS, help="并行下载线程数")
    ap.add_argument("--folders", nargs="*", default=FOLDERS, help="要拉取的远端子目录")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = source_from_env()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 2

    start = time.time()

    # 第一阶段：单连接列文件
    print("正在从服务器列取文件清单...")
    ssh = connect(cfg)
    try:
        sftp = ssh.open_sftp()
        all_files: list[tuple[str, Path, int]] = []
        for folder in args.folders:
            remote = f"{cfg.base_dir}/{folder}"
            local = args.local_root / folder
            files = build_file_list(sftp, remote, local)
            size = sum(f[2] for f in files)
            print(f"  {folder}: {len(files)} 个文件，{size / 1024 / 1024:.1f} MB")
            all_files.extend(files)
        sftp.close()
    finally:
        ssh.close()

    total = len(all_files)
    total_bytes = sum(f[2] for f in all_files)
    if total == 0:
        print("远端没有找到任何文件，请检查 PUNQUIZCN_SRC_BASE 是否正确。")
        return 1
    print(f"\n待下载：{total} 个文件，{total_bytes / 1024 / 1024 / 1024:.2f} GB")

    # 第二阶段：并行下载
    work_queue: queue.Queue[tuple[str, Path, int, int]] = queue.Queue()
    for i, (remote_path, local_path, size) in enumerate(all_files):
        work_queue.put((remote_path, local_path, size, i))

    lock = threading.Lock()
    done_counter = [0]
    failed: list[str] = []

    threads = [
        threading.Thread(
            target=worker,
            args=(cfg, work_queue, total, lock, done_counter, failed),
            daemon=True,
        )
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"\n下载完成，用时 {elapsed / 60:.1f} 分钟。")
    if failed:
        print(f"失败 {len(failed)} 个文件：")
        for path in failed[:50]:
            print("  " + path)
        return 1
    print("全部文件下载成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
