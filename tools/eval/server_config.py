"""远程推理服务配置：从环境变量 / CLI 读取，**不含任何内网地址或密钥**。

本模块把「服务器在哪、怎么起、用哪个模型」这三件事从代码里彻底剥离出来，
换成运行时可配置的形态。所有脚本共用这里的读取逻辑，避免各处硬编码。

环境变量
--------
``PUNQUIZCN_HOST``
    SSH 目标，形如 ``user@gpu-box.example.com``（必填，用于远端起服务）。
``PUNQUIZCN_SSH_KEY``
    SSH 私钥路径（可选；不给则用 ssh 默认的密钥/agent）。
``PUNQUIZCN_BASE``
    推理服务 HTTP 地址，形如 ``http://localhost:15021``。
    本地直接跑服务时用它；不给则回落到 ``http://<HOST 主机名>:15021``。
``PUNQUIZCN_API_KEY``
    Bearer token（可选；llama.cpp 默认不需要）。
``PUNQUIZCN_LLAMA_SERVER``
    远端 llama-server 可执行文件路径（默认 ``$HOME/llama.cpp/build/bin/llama-server``）。
``PUNQUIZCN_PORT``
    推理服务端口（默认 15021）。

模型清单
--------
模型 gguf 路径因机器而异，所以**不写死**。用 JSON 文件声明，格式：

::

    [
      {"tag": "qwen3vl_8b",
       "args": "-m $HOME/models/qwen3vl8b.gguf --mmproj $HOME/models/qwen3vl8b_mmproj.gguf "
               "--ctx-size 32768 --parallel 4"}
    ]

``args`` 是直接拼给 llama-server 的命令行串（远端 shell 展开 ``$HOME`` 等变量）。
用 ``--models-file`` 指向该文件；不给则用 ``--models`` 传内联 JSON。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = 15021
DEFAULT_LLAMA_SERVER = "$HOME/llama.cpp/build/bin/llama-server"


class ConfigError(RuntimeError):
    """配置缺失或非法。故意抛得显眼，避免静默连到错误的服务。"""


@dataclass(frozen=True)
class Remote:
    """远端推理环境。"""

    host: str
    ssh_key: Path | None = None
    port: int = DEFAULT_PORT
    llama_server: str = DEFAULT_LLAMA_SERVER

    def ssh_base(self) -> list[str]:
        """``ssh`` 命令的公共前缀（不含具体命令）。"""
        cmd = ["ssh"]
        if self.ssh_key is not None:
            cmd += ["-i", str(self.ssh_key)]
        cmd += [
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.port if self.port != 22 else 22),
            self.host,
        ]
        return cmd


@dataclass(frozen=True)
class Endpoint:
    """推理服务的 HTTP 入口。"""

    base: str
    api_key: str | None = None

    @property
    def display(self) -> str:
        """用于日志：不打印密钥。"""
        return self.base


@dataclass(frozen=True)
class ModelSpec:
    """一个模型的 llama-server 启动参数。"""

    tag: str
    args: str

    @classmethod
    def from_dict(cls, obj: dict[str, str]) -> ModelSpec:
        try:
            return cls(tag=obj["tag"], args=obj["args"])
        except KeyError as exc:  # pragma: no cover - 配置错误
            raise ConfigError(f"模型条目缺少字段 {exc}：{obj}") from exc


def remote_from_env() -> Remote:
    """从环境变量构造 ``Remote``。"""
    host = os.environ.get("PUNQUIZCN_HOST", "").strip()
    if not host:
        raise ConfigError(
            "未设置 PUNQUIZCN_HOST。请在环境变量里给出 SSH 目标，例如：\n"
            "  $env:PUNQUIZCN_HOST = 'user@gpu-box.example.com'   (PowerShell)\n"
            "  export PUNQUIZCN_HOST=user@gpu-box.example.com     (bash)"
        )
    key = os.environ.get("PUNQUIZCN_SSH_KEY", "").strip()
    port_raw = os.environ.get("PUNQUIZCN_PORT", "").strip()
    return Remote(
        host=host,
        ssh_key=Path(key).expanduser() if key else None,
        port=int(port_raw) if port_raw else DEFAULT_PORT,
        llama_server=os.environ.get("PUNQUIZCN_LLAMA_SERVER", DEFAULT_LLAMA_SERVER),
    )


def endpoint_from_env(default_port: int = DEFAULT_PORT) -> Endpoint:
    """从环境变量构造 ``Endpoint``。

    优先用 ``PUNQUIZCN_BASE``；否则用 ``PUNQUIZCN_HOST`` 的主机名拼一个。
    """
    base = os.environ.get("PUNQUIZCN_BASE", "").strip()
    if not base:
        host = os.environ.get("PUNQUIZCN_HOST", "").strip()
        if not host:
            raise ConfigError(
                "未设置 PUNQUIZCN_BASE，也没有 PUNQUIZCN_HOST 可用于推断。请至少给出其一。"
            )
        hostname = host.rsplit("@", 1)[-1].split(":", 1)[0]
        base = f"http://{hostname}:{default_port}"
    key = os.environ.get("PUNQUIZCN_API_KEY", "").strip()
    return Endpoint(base=base.rstrip("/"), api_key=key or None)


def load_models(path: Path | None, inline: str | None) -> list[ModelSpec]:
    """从 JSON 文件或内联 JSON 串读模型清单。"""
    if path is None and not inline:
        raise ConfigError(
            "未指定模型清单。请用 --models-file 指向 JSON 文件，或用 --models 传内联 JSON 数组。"
        )
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"模型清单文件不存在：{path}")
        raw = path.read_text(encoding="utf-8")
    else:
        raw = inline or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"模型清单不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, list):
        raise ConfigError("模型清单必须是 JSON 数组。")
    return [ModelSpec.from_dict(item) for item in parsed]
