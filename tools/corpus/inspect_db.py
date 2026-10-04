"""检查远端标注库：列表、表结构、行数、样本行、来源分布。

做法是把一段自省脚本上传到远端用 python3 跑一遍，再取回输出。
**远端根目录由配置注入**，不再写死任何用户名或路径。

用法
----
::

    $env:PUNQUIZCN_SRC_HOST = 'anno.example.com'
    $env:PUNQUIZCN_SRC_USER = 'alice'
    $env:PUNQUIZCN_SRC_KEY  = 'C:\\Users\\me\\.ssh\\id_ed25519'
    python tools/corpus/inspect_db.py

环境变量说明见 ``source_config.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from source_config import ConfigError, connect, source_from_env

#: 远端自省脚本。``{db}`` 会被替换成实际数据库路径。
REMOTE_SCRIPT = """
import sqlite3, os, json
db = {db!r}
print("DB size:", round(os.path.getsize(db) / 1024 / 1024 / 1024, 2), "GB")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
print("\\n=== tables ===")
for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(" -", r[0])
print("\\n=== annotations schema ===")
for r in cur.execute("PRAGMA table_info(annotations)"):
    print("  ", r["name"], r["type"])
print("\\n=== row counts ===")
for r in cur.execute("SELECT 'annotations' t, COUNT(*) c FROM annotations"):
    print("  annotations:", r["c"], "rows")
print("\\n=== sample 2 rows ===")
for r in cur.execute("SELECT * FROM annotations LIMIT 2"):
    d = dict(r)
    for k in ("hint_image", "guess_image", "hint_thumb", "guess_thumb"):
        if k in d and d[k]:
            d[k] = str(d[k])[:100] + " ..."
    print(json.dumps(d, ensure_ascii=False, indent=2))
print("\\n=== distinct 'from' (top 40) ===")
for r in cur.execute(
    'SELECT "from", COUNT(*) c FROM annotations GROUP BY "from" ORDER BY c DESC LIMIT 40'
):
    print(f"  {{r['from']}}: {{r['c']}}")
conn.close()
"""

REMOTE_PATH = "/tmp/inspect_annotations_db.py"


def main() -> int:
    try:
        cfg = source_from_env()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 2

    script = REMOTE_SCRIPT.format(db=cfg.db_path())
    print(f"连接 {cfg.host} ...")
    ssh = connect(cfg)
    try:
        sftp = ssh.open_sftp()
        with sftp.open(REMOTE_PATH, "w") as fh:
            fh.write(script)
        print("脚本已上传，开始执行...\n")
        _, stdout, stderr = ssh.exec_command(
            f"python3 {REMOTE_PATH}; rm -f {REMOTE_PATH}", timeout=300
        )
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        print(out)
        if err.strip():
            print("[STDERR]", err[:3000])
        sftp.close()
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
