"""语料源站（标注系统）的连接配置：**全部从环境变量读取，绝不写死**。

这些脚本访问的是标注阶段用来存放 ``annotations.db`` 的服务器。地址、
账号、密码都随部署环境而变，所以一律走环境变量；仓库里不含任何真实凭据。

环境变量
--------
``PUNQUIZCN_SRC_HOST``
    主机名或 IP（必填）。
``PUNQUIZCN_SRC_PORT``
    SSH 端口（默认 22）。
``PUNQUIZCN_SRC_USER``
    用户名（必填）。
``PUNQUIZCN_SRC_PASSWORD``
    密码。与 ``PUNQUIZCN_SRC_KEY`` 二选一。
``PUNQUIZCN_SRC_KEY``
    SSH 私钥路径。与 ``PUNQUIZCN_SRC_PASSWORD`` 二选一（**推荐用密钥**）。
``PUNQUIZCN_SRC_BASE``
    标注系统在远端的根目录（例如 ``/srv/annoSys``）。
    默认取 ``$HOME/project/annoSys`` 的展开值。

安全提示
--------
密码走环境变量仍然会出现在进程环境里。生产环境请优先用 ``PUNQUIZCN_SRC_KEY``
配 SSH 密钥，或改用 ssh-agent。**不要把密码写进任何被版本控制的文件。**
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """配置缺失或非法。故意抛得显眼，避免静默连到错误的服务器。"""


@dataclass(frozen=True)
class SourceConfig:
    """远端标注系统的连接参数。"""

    host: str
    user: str
    password: str | None = None
    key: Path | None = None
    port: int = 22
    base: str = ""

    @property
    def base_dir(self) -> str:
        """远端根目录；未显式配置时用 ``~/project/annoSys``。"""
        return self.base or f"/home/{self.user}/project/annoSys"

    def db_path(self) -> str:
        return f"{self.base_dir}/annotations.db"

    def require_auth(self) -> None:
        """确认至少给了一种认证方式。"""
        if not self.password and self.key is None:
            raise ConfigError(
                "未提供认证方式。请设置 PUNQUIZCN_SRC_PASSWORD 或 PUNQUIZCN_SRC_KEY。\n"
                "推荐用 SSH 密钥：$env:PUNQUIZCN_SRC_KEY = 'C:\\path\\to\\id_rsa'"
            )


def source_from_env() -> SourceConfig:
    """从环境变量构造 ``SourceConfig``。"""
    host = os.environ.get("PUNQUIZCN_SRC_HOST", "").strip()
    user = os.environ.get("PUNQUIZCN_SRC_USER", "").strip()
    missing = [n for n, v in (("PUNQUIZCN_SRC_HOST", host), ("PUNQUIZCN_SRC_USER", user)) if not v]
    if missing:
        raise ConfigError(
            "缺少必需的环境变量：" + "、".join(missing) + "\n"
            "例如：\n"
            "  $env:PUNQUIZCN_SRC_HOST = 'anno.example.com'\n"
            "  $env:PUNQUIZCN_SRC_USER = 'alice'\n"
            "  $env:PUNQUIZCN_SRC_KEY  = 'C:\\Users\\me\\.ssh\\id_ed25519'"
        )
    port_raw = os.environ.get("PUNQUIZCN_SRC_PORT", "").strip()
    key_raw = os.environ.get("PUNQUIZCN_SRC_KEY", "").strip()
    cfg = SourceConfig(
        host=host,
        user=user,
        password=os.environ.get("PUNQUIZCN_SRC_PASSWORD") or None,
        key=Path(key_raw).expanduser() if key_raw else None,
        port=int(port_raw) if port_raw else 22,
        base=os.environ.get("PUNQUIZCN_SRC_BASE", "").strip(),
    )
    cfg.require_auth()
    return cfg


def connect(cfg: SourceConfig):  # type: ignore[no-untyped-def]
    """按配置建立一个 paramiko SSHClient（延迟导入 paramiko）。

    返回已 connect() 的 client。调用方负责 close()。
    """
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ConfigError(
            "这些语料工具需要 paramiko。请安装：uv add --group tools paramiko"
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if cfg.key is not None:
        client.connect(
            cfg.host, port=cfg.port, username=cfg.user, key_filename=str(cfg.key), timeout=60
        )
    else:
        client.connect(
            cfg.host, port=cfg.port, username=cfg.user, password=cfg.password, timeout=60
        )
    return client
