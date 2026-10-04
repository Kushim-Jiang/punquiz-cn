#!/usr/bin/env python3
"""
PunQuizCN 评测提示词套件 / prompt suite for PunQuizCN (Chinese two-image homophone-rebus corpus)

设计文档：paper/11_评测提示词设计.md
版本号：PROMPT_VERSION —— 必须随结果一起落盘，便于论文附录复现。

覆盖任务
--------
T1  双图开放解码（输入档 A/B/C/D × 输出档 answer_only/json × 答案形状提示：字数/大类，默认关）
T1' 后续轮（build_t1_retry）：先照 build_t1 答一轮（那一次算 one-shot），后续轮把
    “上一轮答了什么 + 反馈（含正确效果的符号说明）”追加到 user 侧要求重答。反馈档：
      "none" 控制组（只说错误） / "bin" / "tri" / "phon" / "phon_char"（汉兜四项属性）/ "phon_near"
    ⚠️ 全程 **system 逐字不变**（同一个 _t1_system），其余参数必须与第一轮完全一致。
T4  逐位选字填空（输出档 letters/json）
W   Wordle 式多轮猜测。反馈档 feedback：
      "tri"       经典字级三态（🟩位对 / 🟨位错 / ⬜无）
      "bin"       字级二态（只报位对）
      "phon"      音系三项属性：声母 / 韵母 / 声调
      "phon_char" 音系三项属性 + 字属性（= 汉兜 Handle 的四属性反馈）
      "phon_near" 同 phon_char，但把 zh/z、ch/c、sh/s、n/l、r/l、f/h 与前后鼻音视为相近
    另可调：轮数 / 是否告知字数 / 是否给候选字池

调用协议
--------
OpenAI 兼容的 /v1/chat/completions：messages 列表；图片以 {"type":"image_url"} 内嵌
data:image/jpeg;base64,... 传入。**每次请求无状态：多轮也在每轮重发两张图**，
避免不同服务端会话记忆策略带来的口径差异。

用法
----
    from prompts import build_t1, build_t4, build_wordle_turn, WordleConfig
    msgs = build_t1(item, input_mode="A", output_mode="json")     # item 来自 benchmark_t1.jsonl
    msgs = build_t1_retry(item, prev_answers=["..."], feedback="phon_char")   # 后续轮（system 同第一轮）
    msgs = build_t4(item, output_mode="letters")                  # item 来自 benchmark_t4.jsonl
    cfg  = WordleConfig(max_turns=6, reveal_length=True, feedback="tri")
    msgs = build_wordle_turn(item, history=[], cfg=cfg, turn=1)   # history: 历史猜测字符串列表

安全约定
--------
- T4 的 gold 字段**绝不进入提示词**（benchmark_t4.jsonl 的 positions[].gold 只用于判分）。
- Wordle 的 answer 只用于服务端算反馈，绝不渲染进 prompt。
- **提示词里举例用的词必须是人造词，绝不能是语料里的真实答案**（曾误用「拿破仑」「金枪鱼」，两次都是泄漏）。
  统一定义在 _PROMPT_EXAMPLE_WORDS，_demo() 里有断言对着语料答案集守着。
- 后续轮的反馈由服务端算，**gold 绝不渲染进 prompt**；`_demo()` 有断言盯着。
  多轮全程 **system 逐字不变**，否则第一轮的 one-shot 不再干净（同样有断言）。

依赖
----
- 字级反馈（tri/bin）与 T1/T4 不需要额外依赖。
- 音系反馈（phon/phon_char/phon_near）需要 `pypinyin`，本项目已列为依赖
  （见 pyproject.toml），`uv sync` 会自动装好。
  `phon_marks()` 在缺库时会直接 raise，不会静默给出全灰反馈。
"""

from __future__ import annotations

import base64
import contextlib
import csv
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPT_VERSION = "pzcn-prompt-v1.2"

# ---------------------------------------------------------------------------
# 语料定位
# ---------------------------------------------------------------------------
# 完整语料不随代码发布（体积原因）。若使用者手上有语料，可通过环境变量
# PUNQUIZCN_CORPUS 指向 release_corpus_final.tsv，或把它放到本包 data/ 下。
_CORPUS_ENV = "PUNQUIZCN_CORPUS"


def corpus_path() -> Path | None:
    """返回语料 TSV 的路径；找不到时返回 None（调用方应优雅跳过）。

    查找顺序：
      1. 环境变量 ``PUNQUIZCN_CORPUS``
      2. ``<repo>/data/release_corpus_final.tsv``
      3. ``<repo>/data/sample/sample_items.jsonl`` 所在目录的兄弟 TSV 位置

    第 2、3 项都相对于本文件向上回溯到仓库根，避免依赖当前工作目录。
    """
    import os

    env = os.environ.get(_CORPUS_ENV)
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None

    # prompts.py -> eval -> punquizcn -> src -> <repo root>
    repo = Path(__file__).resolve().parents[3]
    for cand in (
        repo / "data" / "release_corpus_final.tsv",
        repo / "data" / "release" / "release_corpus_final.tsv",
    ):
        if cand.is_file():
            return cand
    return None


# ============================================================================
# 0. 归一化与判分
# ============================================================================
# 语料把「答案」定义为**读音相同的词**，所以同音异字也算对。判分因此分三级，
# 全部建立在同一套音系分解之上（见 §5 的算法化描述）。


def norm(s: str) -> str:
    """只保留汉字、字母、数字；去掉空白与标点。"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s or "")


def _syllables(text: str) -> list[str]:
    """逐字音节（带调，TONE3 形式，如 ``shi4``）。缺 pypinyin 时抛错。"""
    from pypinyin import Style, pinyin

    return ["".join(x) for x in pinyin(text, style=Style.TONE3, errors="ignore")]


def is_correct(guess: str, gold: str) -> tuple[bool, str]:
    """判分：语料定义答案 = 读音相同的词，所以同音异字也算对。

    返回 ``(对否, 判分依据)``。三级：

    ``exact``
        汉字逐字相同。
    ``homophone``
        逐位音节（含声调）完全相同，只是写了别的字。
    ``near``
        逐位音节去掉声调后相同（近音，单独标记，不计入主口径）。
    """
    g, d = norm(guess), norm(gold)
    if not g:
        return False, "empty"
    if g == d:
        return True, "exact"
    gs, ds = _syllables(g), _syllables(d)
    if len(gs) == len(ds) and gs and gs == ds:
        return True, "homophone"
    if (
        len(gs) == len(ds)
        and gs
        and [s.rstrip("012345") for s in gs] == [s.rstrip("012345") for s in ds]
    ):
        return True, "near"
    return False, "no"


# ============================================================================
# 0b. 提示词泄漏护栏
# ============================================================================
# 硬护栏：提示词里绝不能出现答案本身或其任何一位的读音。
#
# 三道防线：
#   1) 答案汉字串不能被原样包含；
#   2) 答案逐位音节带调形式不能被包含（``shan1``）；
#   3) 答案逐位的声母/韵母/声调被**拆开写明**也算泄露（"声母 sh 韵母 an 声调 1"）。
#
# 两类必须排除的误杀（都踩过）：
#   * 多轮协议会把**模型自己上一轮的答案**写进提示词。模型自己猜中了，
#     那段文本里当然会有 gold——这是合法且必然发生的。
#   * 答案恰好是**提示词模板自身的常用词**（如答案「代码」被模板里
#     "不要用 ``` 代码块包裹"命中）。模板本来就要给模型看，不构成泄露。
#     因此检查前先把模板自带的固定文本整段剥掉，只查动态注入的部分。


def flatten_messages(messages: Sequence[dict[str, Any]]) -> str:
    """把 messages 里所有文本拍平成字符串（图片 data URL 不计入）。"""
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        else:
            for blk in content or []:
                if blk.get("type") == "text":
                    out.append(str(blk.get("text", "")))
    return "\n".join(out)


@lru_cache(maxsize=1)
def _boilerplate_strings() -> tuple[str, ...]:
    """收集本模块里所有模板常量（str 类型、长度 >= 8），供剥除用。

    长的先替换，避免短串把长串切碎。
    """
    out: list[str] = []
    for name in dir(sys.modules[__name__]):
        if name.startswith("_"):
            continue
        value = globals().get(name)
        if isinstance(value, str) and len(value) >= 8:
            out.append(value)
    out.sort(key=len, reverse=True)
    return tuple(out)


def strip_prompt_boilerplate(text: str) -> str:
    """把提示词模板自带的固定文本从待检串里剥掉。"""
    out = text
    for s in _boilerplate_strings():
        if s:
            out = out.replace(s, "")
    return out


def assert_no_leak(
    messages: Sequence[dict[str, Any]],
    item: dict[str, Any],
    scheme: str,
    turn: int,
    prev_answers: Sequence[str] | None = None,
) -> None:
    """硬护栏：提示词里不得出现答案本身或其任何一位的读音。违者抛 AssertionError。"""
    text = flatten_messages(messages)
    gold = norm(str(item.get("answer", "")))
    scrubbed = strip_prompt_boilerplate(text)

    # 剥掉「模型自己的历次回答」——它猜中时必然含 gold，那不是泄露
    for a in prev_answers or []:
        na = norm(str(a))
        if na:
            scrubbed = scrubbed.replace(na, "")

    if gold and len(gold) >= 2 and gold in scrubbed:
        raise AssertionError(
            f"[L{item.get('id')}] 方案 {scheme} 第{turn}轮：提示词中出现了答案「{gold}」！"
        )

    flat = scrubbed.replace(" ", "").replace("\u3000", "")
    for syl in _syllables(gold):
        if len(syl) < 3:
            continue
        if syl in scrubbed or syl in flat:
            raise AssertionError(
                f"[L{item.get('id')}] 方案 {scheme} 第{turn}轮：提示词中出现了答案读音「{syl}」"
                "（等价于报答案）！"
            )

    # 第 3 条：拆成属性写出来也算泄露。逐位检查 gold 的 (韵母, 声调)
    # 是否在文中以「韵母…声调数字」的形式出现。
    from pypinyin import Style, pinyin

    finals = ["".join(x) for x in pinyin(gold, style=Style.FINALS, strict=True, errors="ignore")]
    tones = ["".join(x) for x in pinyin(gold, style=Style.TONE3, errors="ignore")]
    for i in range(len(gold)):
        fin = finals[i] if i < len(finals) else ""
        tone = tones[i][-1] if i < len(tones) and tones[i] and tones[i][-1].isdigit() else ""
        if not fin or not tone:
            continue
        if (fin in scrubbed or fin in flat) and re.search(
            re.escape(fin) + r"\D{0,6}" + re.escape(tone), text
        ):
            raise AssertionError(
                f"[L{item.get('id')}] 方案 {scheme} 第{turn}轮：提示词拆开写出了答案读音"
                f"（韵母 {fin} + 声调 {tone}）——等价于报答案！"
            )


# ============================================================================
# 0c. 图片编码
# ============================================================================
DEFAULT_MAX_SIDE = 1280
DEFAULT_JPEG_QUALITY = 85


def encode_image_data_url(
    path: str | Path,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """把本地图片编码成 data URL（长边缩到 max_side，JPEG 重编码控 token）。

    若 PIL 不可用或解码失败，退回原文件直读 base64。
    """
    p = Path(path)
    try:
        from PIL import Image

        with Image.open(p) as src:
            im = src.convert("RGB")
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                im = im.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = buf.getvalue()
    except Exception:  # noqa: BLE001 - PIL 缺失/解码失败都退回原始字节，不该让一张坏图中断整轮评测
        raw = p.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _img_part(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _user_content(text: str, data_urls: Sequence[str]) -> Any:
    """有图 → 多模态 content 列表；无图 → 纯字符串（兼容性更好）。"""
    if not data_urls:
        return text
    return [_text_part(text)] + [_img_part(u) for u in data_urls]


# ============================================================================
# 1. T1 —— 双图开放解码
# ============================================================================

T1_ROLE_ZH = "你是一个中文谐音谜题解题者。"

T1_PUZZLE_IMAGES_ZH = """谜题形式：给出两张卡片图片，两卡是同一道谜题的两条线索，要结合起来看。
- 图1（提示卡）：画面是一个物体，卡面上通常印有“这是X”，“X”就是该物体的名字。
- 图2（猜测卡）：画面是另一个场景或动作，卡面提示“这是___”——图2的画面通常就是把答案的字面意思画出来。
解题方式：结合两图的关系，说出图2画面所对应的那个谐音梗——一个常见的中文词/成语/人名/地名，其读音与图2画面相谐（同音或近音）。"""

T1_PUZZLE_TEXT_ZH = """谜题形式：这道题原本由两张卡片组成，现以文字给出两卡的说明，两卡要结合起来看。
- 图1（提示卡）的说明：{hint}——给出了画面中一个物体的名字。
- 图2（猜测卡）的画面描述：{guess}——图2的画面通常就是把答案的字面意思画出来。
解题方式：结合两图的关系，说出图2画面所对应的那个谐音梗——一个常见的中文词/成语/人名/地名，其读音与图2画面相谐（同音或近音）。"""

T1_NOTE_B_ZH = "补充信息：图1卡面上印的是“{hint}”。除卡面文字与两张图外没有其他线索。"

T1_NOTE_C_ZH = "补充信息：图1卡的说明：{hint}；图2卡的画面描述：{guess}。"

# 答案形状提示（额外提示档，默认关；开了会显著降低难度，必须当独立消融档报告）
T1_HINT_LEN_ZH = "补充信息：答案共 {n} 个字。"
T1_HINT_TYPE_ZH = "补充信息：答案是{type_name}。"
T1_HINT_BOTH_ZH = "补充信息：答案是{type_name}，共 {n} 个字。"

#: 答案类别（粗细两级）。数据列存 code（ASCII、稳定），提示词用 hint_type_label() 渲染。
#: 语料**没有**这一列，需一次标注后冻结为 answer_type（判定规则/优先级见 paper/11 §3.3）。
#:   - 细类：给模型的提示用，最具体；
#:   - grp_* 粗类：统计汇总用；也可当 hint_type 传入（= 更弱的提示，即"提示粒度"这一维消融）。
T1_HINT_TYPES: dict[str, tuple[str, str]] = {
    # —— 人物 ——
    "person": ("人名", "a person name"),
    "character": ("角色", "a fictional character name"),
    # —— 地名 ——
    "place": ("地名", "a place name"),
    # —— 品牌／作品／机构 ——
    "brand": ("品牌", "a brand name"),
    "group": ("组合", "a band or group name"),
    "work": ("作品", "a work title"),
    "org": ("机构", "an organisation name"),
    # —— 生物 ——
    "animal": ("动物", "an animal"),
    "plant": ("植物", "a plant"),
    # —— 食物 ——
    "food": ("食物", "a food or drink"),
    # —— 物品（按功能域拆）——
    "furniture": ("家具", "a piece of furniture"),
    "appliance": ("家电", "a home appliance"),
    "kitchenware": ("厨具", "a kitchen utensil or tableware"),
    "device": ("数码", "a digital device"),
    "tool": ("工具", "a tool"),
    "clothing": ("服饰", "an item of clothing"),
    "daily": ("日用品", "a household item"),
    # —— 专有名词（按领域拆）——
    "tech": ("科技词", "a technology or internet term"),
    "science": ("学科词", "a scientific concept"),
    "job": ("职业", "a job title"),
    "sport": ("体育词", "a sports or game term"),
    # —— 网络热词／口语（按功能拆）——
    "netmeme": ("网络梗", "an internet meme or buzzword"),
    "colloquial": ("口语", "a colloquial word"),
    "blessing": ("祝福语", "a blessing or auspicious phrase"),
    "service": ("服务", "a service or activity"),
    # —— 固定短语 ——
    "idiom": ("成语", "a Chinese idiom"),
    "saying": ("俗语", "a common saying"),
    # —— 兜底 ——
    "other": ("其他", "another kind of word"),
    # —— 粗类（只用于汇总；当 hint_type 传入时是更弱的提示）——
    "grp_person": ("人名或角色", "a person or character name"),
    "grp_org": ("品牌或作品", "a brand, work, group or organisation name"),
    "grp_object": ("物品", "a common object or appliance"),
    "grp_term": ("专有名词", "a technical or domain-specific term"),
    "grp_netword": ("网络热词", "a colloquial or internet word"),
    "grp_phrase": ("固定短语", "a set phrase"),
}

#: 细类 → 粗类（粗类自身映射到自身 code，便于统一 roll-up）。
T1_HINT_PARENT: dict[str, str] = {
    "person": "grp_person",
    "character": "grp_person",
    "place": "place",
    "brand": "grp_org",
    "group": "grp_org",
    "work": "grp_org",
    "org": "grp_org",
    "animal": "animal",
    "plant": "plant",
    "food": "food",
    "furniture": "grp_object",
    "appliance": "grp_object",
    "kitchenware": "grp_object",
    "device": "grp_object",
    "tool": "grp_object",
    "clothing": "grp_object",
    "daily": "grp_object",
    "tech": "grp_term",
    "science": "grp_term",
    "job": "grp_term",
    "sport": "grp_term",
    "netmeme": "grp_netword",
    "colloquial": "grp_netword",
    "blessing": "grp_netword",
    "service": "grp_netword",
    "idiom": "grp_phrase",
    "saying": "grp_phrase",
    "other": "other",
}


def hint_type_label(code_or_name: str, lang: str = "zh") -> str:
    """把类别 code（细类或 grp_* 粗类，或已写好的说法）转成提示词里要写的字符串。

    hint_type_label("animal") → 中文 "动物" / 英文 "an animal"；
    hint_type_label("grp_object") → "物品"（更弱的提示）；
    不在表里的字符串原样返回（允许调用方自定义，如 "常见词语"）。
    """
    hit = T1_HINT_TYPES.get(code_or_name)
    if not hit:
        return code_or_name
    return hit[0] if lang == "zh" else hit[1]


def hint_type_parent(code: str) -> str:
    """细类 → 粗类 code（未知 code 原样返回）；统计汇总用。"""
    return T1_HINT_PARENT.get(code, code)


T1_JSON_ZH = """输出要求：只输出一个 JSON 对象，不要输出任何解释文字，不要用 ``` 代码块包裹，不要在 JSON 前后添加内容。字段如下：
{
  "hint_name": "图1所指对象的名字（2-4字）",
  "guess_name": "图2画面所指对象或动作的名字（短语）",
  "hint_pinyin": "hint_name 的拼音，可带声调",
  "guess_pinyin": "guess_name 的拼音，可带声调",
  "answer": "最终答案"
}
"answer" 只写一个答案，必须是常见的中文词/成语/人名/地名（个别题含字母或数字则按原样）；无法确定时给出最可能的答案，不要留空、不要写“无法确定”。"""

T1_ANSWER_ONLY_ZH = """输出要求：只输出答案本身（一个词/成语/人名/地名），不要输出拼音、解释、序号或标点，不要用 ``` 代码块包裹。只含汉字（个别题含字母或数字则按原样）。"""

T1_ROLE_EN = "You are a solver of Chinese homophone rebus puzzles."

T1_PUZZLE_IMAGES_EN = """Puzzle format: two card images are given; together they are the two clues of one puzzle and must be read jointly.
- Card 1 (hint card): depicts an object and usually prints "this is X", where X is the object's name.
- Card 2 (guess card): depicts another scene or action, printed as "this is ___"; this scene usually draws the answer's literal meaning.
How to solve: read the two images together and produce the pun that the second image depicts - a common Chinese word / idiom / person name / place name whose sound matches the scene in card 2 (homophone or near-homophone)."""

T1_PUZZLE_TEXT_EN = """Puzzle format: the puzzle originally consists of two cards; both are given as text and must be read jointly.
- Card 1 (hint card) description: {hint} - names an object that appears in the scene.
- Card 2 (guess card) scene description: {guess} - this scene usually draws the answer's literal meaning.
How to solve: read the two images together and produce the pun that the second image depicts - a common Chinese word / idiom / person name / place name whose sound matches the scene in card 2 (homophone or near-homophone)."""

T1_NOTE_B_EN = 'Extra information: card 1 prints "{hint}". There are no further clues beyond the card text and the two images.'

T1_NOTE_C_EN = "Extra information: card 1 description: {hint}; card 2 scene description: {guess}."

T1_HINT_LEN_EN = "Extra information: the answer has {n} characters."
T1_HINT_TYPE_EN = "Extra information: the answer is {type_name}."
T1_HINT_BOTH_EN = "Extra information: the answer is {type_name} with {n} characters."


def _t1_answer_hint(lang: str, n: int | None = None, type_name: str | None = None) -> str:
    """答案形状提示：字数 / 大类。两者都不给 → 空串。

    口径：这是**额外提示**，与输入档 A/B/C/D 正交（可叠加），默认关闭。
    打开后难度会明显下降，必须作为单独的消融档报告，不能与零提示档混在一张表里。

    type_name 建议直接传 T1_HINT_TYPES 的 code（如 "animal"/"idiom"），会自动渲染成
    中文/英文说法；也可传自定义字符串（则原样使用，需自己保证语言匹配）。
    注意：本语料**没有**现成的"答案大类"列（只有 has_latin/has_digit），
    需按 paper/11 §3.2 的规则标注一次后冻结。
    """
    if n is None and not type_name:
        return ""
    zh = lang == "zh"
    label = hint_type_label(type_name, lang) if type_name else ""
    if n is not None and label:
        return (T1_HINT_BOTH_ZH if zh else T1_HINT_BOTH_EN).format(n=n, type_name=label)
    if n is not None:
        return (T1_HINT_LEN_ZH if zh else T1_HINT_LEN_EN).format(n=n)
    return (T1_HINT_TYPE_ZH if zh else T1_HINT_TYPE_EN).format(type_name=label)


T1_JSON_EN = """Output requirements: output exactly one JSON object. No explanation, no ``` fences, nothing before or after the JSON. Fields:
{
  "hint_name": "name of the object on card 1 (2-4 Chinese characters)",
  "guess_name": "name of the object/action depicted on card 2 (a short phrase)",
  "hint_pinyin": "pinyin of hint_name, tones optional",
  "guess_pinyin": "pinyin of guess_name, tones optional",
  "answer": "the final answer"
}
"answer" must be exactly one common Chinese word / idiom / person name / place name (if the item uses letters or digits, keep them as-is). If unsure, give the most likely answer; never leave it empty and never write "unknown"."""

T1_ANSWER_ONLY_EN = """Output requirements: output only the answer itself (one word / idiom / person name / place name). No pinyin, no explanation, no numbering, no punctuation, no ``` fences. Chinese characters only (keep letters or digits as-is when the item contains them)."""

#: 输入档 → (是否带图, 附加句模板键)
T1_INPUT_MODES = ("A", "B", "C", "D")
T1_OUTPUT_MODES = ("answer_only", "json")

#: 首条消息的角色。
#: 为什么可切：Gemma 4 的 chat template 里有这么一行——
#:     {%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system','developer']) -%}
#: 只要**首条消息是 system 角色**，thinking 就被强制激活。thinking 一旦激活，
#: 模型的推理 token 会和答案抢同一个 max_tokens 预算，实测出现
#: “content 为空 + finish_reason=length”的截断。把首条消息换成 user 角色
#: 可以彻底绕开这条分支（连 thinking 都不激活）。
#: 代价：system 角色的约束力略强于 user，所以两条都要能跑、当独立消融档报告。
T1_ROLE_MODES: tuple[str, ...] = ("system", "user")


def _t1_system(
    item: dict[str, Any],
    input_mode: str = "A",
    output_mode: str = "json",
    lang: str = "zh",
    hint_length: bool | int | None = None,
    hint_type: str | None = None,
) -> str:
    """构造 T1 的 system 提示词。

    **第一轮与后续轮（build_t1_retry）必须用同一个函数产出的逐字相同文本**，
    否则第二轮等于中途换了任务框架，第一轮的回答就不再是干净的 one-shot 了。
    所以这一段固定（包含“答案共 N 个字”这类形状提示）随每次请求整体重发。
    """
    if input_mode not in T1_INPUT_MODES:
        raise ValueError(f"input_mode must be one of {T1_INPUT_MODES}")
    if output_mode not in T1_OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {T1_OUTPUT_MODES}")
    zh = lang == "zh"
    hint = item.get("hint_text", "")
    guess = item.get("guess_text", "")

    has_images = input_mode != "D"
    if has_images:
        puzzle = T1_PUZZLE_IMAGES_ZH if zh else T1_PUZZLE_IMAGES_EN
    else:
        tpl = T1_PUZZLE_TEXT_ZH if zh else T1_PUZZLE_TEXT_EN
        puzzle = tpl.format(hint=hint, guess=guess)

    parts = [T1_ROLE_ZH if zh else T1_ROLE_EN, puzzle]
    if has_images and input_mode == "B":
        parts.append((T1_NOTE_B_ZH if zh else T1_NOTE_B_EN).format(hint=hint))
    if has_images and input_mode == "C":
        parts.append((T1_NOTE_C_ZH if zh else T1_NOTE_C_EN).format(hint=hint, guess=guess))

    # 答案形状提示（字数 / 大类）——正交提示档
    n_hint: int | None = None
    if hint_length is True:
        n_hint = len(str(item.get("answer", "")))
    elif isinstance(hint_length, int) and not isinstance(hint_length, bool):
        n_hint = int(hint_length)
    hint_txt = _t1_answer_hint(lang, n=n_hint, type_name=hint_type)
    if hint_txt:
        parts.append(hint_txt)

    parts.append(
        T1_JSON_ZH
        if output_mode == "json" and zh
        else T1_JSON_EN
        if output_mode == "json"
        else T1_ANSWER_ONLY_ZH
        if zh
        else T1_ANSWER_ONLY_EN
    )
    return "\n\n".join(parts)


#: T1 后续轮的反馈档（除 "none" 外都直接复用 Wordle 那套 render_feedback / _phon_legend）：
#:   none       控制组：只说“错误，再猜”，不给任何信息（用于分离“多给一次机会”本身的收益）
#:   bin        字级二态（只报位对）
#:   tri        字级三态（🟩位对 / 🟨字在但位错 / ⬜无此字）
#:   phon       音系三项属性：声母/韵母/声调
#:   phon_char  音系三项属性 + 字属性 = 汉兜 Handle 的四属性
#:   phon_near  同 phon_char，但 zh/z、ch/c、sh/s、n/l、r/l、f/h 与前后鼻音视为相近
T1_RETRY_FEEDBACKS: tuple[str, ...] = ("none", "bin", "tri", "phon", "phon_char", "phon_near")


def build_t1(
    item: dict[str, Any],
    input_mode: str = "A",
    output_mode: str = "json",
    lang: str = "zh",
    image_encoder: Callable[[str], str] = encode_image_data_url,
    hint_length: bool | int | None = None,
    hint_type: str | None = None,
    role_mode: str = "system",
) -> list[dict[str, Any]]:
    """构造 T1（双图开放解码）**第一轮**的 messages。

    ⚠️ 注意本函数 `hint_length` 默认 `None`，而 build_t1_retry 默认 `True`。
       两者刻意不同（首轮不露字数、后续轮补上），但这意味着**用默认值调两轮
       会导致 system 不一致**。多轮实验必须显式传同一个 hint_length，
       否则第一轮的 one-shot 就被污染了（自检 `T1 retry self-test` 盯的是
       显式 `hint_length=True` 的情形）。

    item 字段：hint_img, guess_img, hint_text, guess_text（answer 不参与构造，除非 hint_length=True）
    input_mode:
        A = 只给两张图（与真实玩法一致）
        B = 两张图 + 图1卡面文字标签
        C = 两张图 + 两卡描述 H'/G'（视觉命名被让出）
        D = 纯文本对照（无图，只给 H'/G'）
    output_mode: "json"（CoT 字段化）| "answer_only"
    hint_length: None = 不给字数提示（默认）；True = 用 len(item["answer"])；int = 显式指定
    hint_type:   如 "成语"/"人名"/"地名"/"网络热词"（None = 不给）
    后两个参数与 input_mode 正交，是可叠加的提示档；开了必须当独立消融档报告。
    role_mode:   "system"（默认，与历史实验可比）| "user"（绕开 Gemma 4 的
                 “system 角色自动开启 thinking”，见 T1_ROLE_MODES 注释）。

    后续轮用 build_t1_retry()，**参数必须与本轮完全一致**（system 要逐字相同）。
    """
    if role_mode not in T1_ROLE_MODES:
        raise ValueError(f"role_mode must be one of {T1_ROLE_MODES}")
    system = _t1_system(item, input_mode, output_mode, lang, hint_length, hint_type)
    zh = lang == "zh"
    has_images = input_mode != "D"

    if has_images:
        ask = (
            "第一张图 = 图1（提示卡），第二张图 = 图2（猜测卡）。请解出这道谐音谜题。"
            if zh
            else "The first image is card 1 (hint card); the second is card 2 (guess card). Solve this puzzle."
        )
        urls = [image_encoder(item["hint_img"]), image_encoder(item["guess_img"])]
    else:
        ask = "请解出这道谐音谜题。" if zh else "Solve this puzzle."
        urls = []

    return [
        {"role": role_mode, "content": system},
        {"role": "user", "content": _user_content(ask, urls)},
    ]


def build_t1_retry(
    item: dict[str, Any],
    prev_answers: Sequence[str],
    feedback: str = "none",
    input_mode: str = "A",
    output_mode: str = "json",
    lang: str = "zh",
    image_encoder: Callable[[str], str] = encode_image_data_url,
    hint_length: bool | int | None = True,
    hint_type: str | None = None,
    gold_phon: Sequence[tuple[str, str, str]] | None = None,
    role_mode: str = "system",
) -> list[dict[str, Any]]:
    """构造 T1 的**后续轮**（第二/三…轮）的 messages。

    与 build_t1 的关系：**system 逐字相同**（同一个 _t1_system），所以“答案共 N 个字”
    这类形状提示随 system 一起重发，每一轮都在。差别只在本轮 user 侧：重发两张图，
    然后逐轮列出**“你答了什么” + “反馈是什么（含正确效果的符号说明）”**，再要求重答。

    prev_answers: 按轮次排列的历次回答（至少 1 个）
    feedback:     T1_RETRY_FEEDBACKS 之一（六档说明见该常量）
    gold_phon:    多音字答案冻结读音，传给音系档用（同 WordleConfig.gold_phon）

    ⚠️ 除 prev_answers/feedback 外，其余参数必须与第一轮 build_t1 传**完全相同的值**，
       否则 system 就不一致，第一轮的 one-shot 也不再干净（自检里有断言盯着）。
    """
    if feedback not in T1_RETRY_FEEDBACKS:
        raise ValueError(f"feedback must be one of {T1_RETRY_FEEDBACKS}")
    if role_mode not in T1_ROLE_MODES:
        raise ValueError(f"role_mode must be one of {T1_ROLE_MODES}")
    if not prev_answers:
        raise ValueError("prev_answers 不能为空；第一轮请用 build_t1()")
    zh = lang == "zh"
    gold = str(item.get("answer", ""))
    n = len(gold)
    cfg = None if feedback == "none" else WordleConfig(feedback=feedback)

    lines: list[str] = [
        "你之前的回答与反馈（按轮次）："
        if zh
        else "Your previous answers and the feedback for each:"
    ]
    for i, ans in enumerate(prev_answers, 1):
        lines.append(f"第{i}轮你的回答：「{ans}」" if zh else f'Round {i} - your answer: "{ans}"')
        if cfg is None:
            # none 档：不给任何结构化信息，只说“错了，再猜”
            lines[-1] += "——错误，再猜。" if zh else " - wrong, guess again."
        else:
            lines.append(
                render_feedback(ans, gold, cfg, lang=lang, gold_phon=gold_phon)["rendered"]
            )
            if len(ans) != n:
                lines.append(
                    f"（注意：这次回答是 {len(ans)} 个字，答案是 {n} 个字。）"
                    if zh
                    else f"(Note: that answer had {len(ans)} characters; the answer has {n}.)"
                )

    if cfg is not None:
        lines += [
            "",
            "反馈说明（上面每行标记是什么意思）：" if zh else "How to read the feedback above:",
            (
                _phon_legend(cfg, lang=lang)
                if feedback in PHON_FEEDBACK_MODES
                else (W_LEGEND_TRI_ZH if zh else W_LEGEND_TRI_EN)
            ),
        ]

    lines += [
        "",
        (
            f"请再回答一次（这是第 {len(prev_answers) + 1} 轮；答案仍是 {n} 个字）。"
            if zh
            else f"Answer again (this is round {len(prev_answers) + 1}; the answer still has {n} characters)."
        ),
    ]
    ask = "\n".join(lines)

    urls: list[str] = []
    if input_mode != "D" and item.get("hint_img") and item.get("guess_img"):
        urls = [image_encoder(item["hint_img"]), image_encoder(item["guess_img"])]
        head = (
            "第一张图 = 图1（提示卡），第二张图 = 图2（猜测卡）。\n"
            if zh
            else "The first image is card 1 (hint card); the second is card 2 (guess card).\n"
        )
        ask = head + ask

    return [
        {
            "role": role_mode,
            "content": _t1_system(item, input_mode, output_mode, lang, hint_length, hint_type),
        },
        {"role": "user", "content": _user_content(ask, urls)},
    ]


# ============================================================================
# 2. T4 —— 逐位选字填空
# ============================================================================

T4_ROLE_ZH = "你是一个中文谐音谜题解题者。"

T4_BODY_ZH = """谜题形式：给出两张卡片图片（图1提示卡印“这是X”，图2猜测卡提示“这是___”），并已按顺序给出答案每个「位」的候选字（每行一个位置）。
你的任务：为每一位从该位的候选字中选 1 个字，按位序拼成答案。
规则：
- 每一位必须从该位给出的候选字中选取，不得使用候选集之外的字。
- 位数与顺序固定，不得增删、调换或漏选。
- 候选字的读音为该位目标音节的同音/近音字，因此只能靠谜面判断。
- 不要输出推理过程、拼音或解释。"""

T4_OUT_LETTERS_ZH = (
    """输出要求：只输出最终答案本身（把每位选中的字依次拼接，不含空格、标点、序号、代码块）。"""
)

T4_OUT_JSON_ZH = """输出要求：只输出一个 JSON 对象，不要用 ``` 代码块包裹，不要有任何多余文字：
{"choices": ["第一位选中的字", "第二位选中的字", "..."]}"""

T4_ROLE_EN = "You are a solver of Chinese homophone rebus puzzles."

T4_BODY_EN = """Puzzle format: two card images are given (card 1 hint card prints "this is X"; card 2 guess card prints "this is ___"), and the candidate characters for each position of the answer are listed in order (one line per position).
Your task: pick exactly one character per position from that position's candidates, and concatenate them in order to form the answer.
Rules:
- Each position must be filled with a character from that position's candidate list; no outside characters.
- The number and order of positions are fixed: do not add, drop, reorder or skip.
- The candidates for a position are homophones/near-homophones of the target syllable there, so only the puzzle itself can disambiguate.
- Do not output reasoning, pinyin or explanations."""

T4_OUT_LETTERS_EN = """Output requirements: output only the answer itself (the selected characters concatenated, no spaces, punctuation, numbering or code fences)."""

T4_OUT_JSON_EN = """Output requirements: output exactly one JSON object, no ``` fences, nothing else:
{"choices": ["char for position 1", "char for position 2", "..."]}"""

T4_OUTPUT_MODES = ("letters", "json")


def render_candidates(positions: Sequence[dict[str, Any]], lang: str = "zh") -> str:
    """把 positions 渲染成“每行一位置”的候选字块。

    ⚠️ 只用 syllable 与 candidates；positions[].gold 绝不能进入提示词。
    """
    lines: list[str] = []
    for i, pos in enumerate(positions, 1):
        cands = " ".join(pos.get("candidates", []))
        syl = pos.get("syllable", "")
        if lang == "zh":
            lines.append(f"第{i}位（读音 {syl}）：{cands}")
        else:
            lines.append(f"position {i} (syllable {syl}): {cands}")
    return "\n".join(lines)


def build_t4(
    item: dict[str, Any],
    output_mode: str = "letters",
    lang: str = "zh",
    image_encoder: Callable[[str], str] = encode_image_data_url,
) -> list[dict[str, Any]]:
    """构造 T4（逐位选字填空）的 messages。item 来自 benchmark_t4.jsonl。"""
    if output_mode not in T4_OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {T4_OUTPUT_MODES}")
    zh = lang == "zh"
    positions = item.get("positions", [])
    n = len(positions)
    system = "\n\n".join(
        [
            T4_ROLE_ZH if zh else T4_ROLE_EN,
            T4_BODY_ZH if zh else T4_BODY_EN,
            (
                T4_OUT_JSON_ZH
                if output_mode == "json" and zh
                else T4_OUT_JSON_EN
                if output_mode == "json"
                else T4_OUT_LETTERS_ZH
                if zh
                else T4_OUT_LETTERS_EN
            ),
        ]
    )
    block = render_candidates(positions, lang=lang)
    if zh:
        ask = (
            f"第一张图 = 图1（提示卡），第二张图 = 图2（猜测卡）。\n"
            f"答案共 {n} 位，各位候选字如下：\n{block}\n\n请给出答案。"
        )
    else:
        ask = (
            f"The first image is card 1 (hint card); the second is card 2 (guess card).\n"
            f"The answer has {n} positions; candidates:\n{block}\n\nGive the answer."
        )
    urls = []
    if item.get("hint_img") and item.get("guess_img"):
        urls = [image_encoder(item["hint_img"]), image_encoder(item["guess_img"])]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _user_content(ask, urls)},
    ]


# ============================================================================
# 3. W —— Wordle 式多轮猜测
# ============================================================================

GREEN, YELLOW, GRAY = "green", "yellow", "gray"
GRID_CHAR = {GREEN: "🟩", YELLOW: "🟨", GRAY: "⬜"}

#: 纯 ASCII 渲染档（供论文/日志等不能用 emoji 的场合）。
#: 语义一一对应 GRID_CHAR / _ATTR_SYM，只是把方形符号换成字母：
#:   G 命中且位置正确 / Y 命中但位置不对 / . 答案中没有 / - 该位不适用
GRID_CHAR_ASCII = {GREEN: "G", YELLOW: "Y", GRAY: "."}

#: 提示词里用于演示例子的**人造词**，与任何真实题目无关。
#: ⚠️ 绝不能是语料里的真实答案（曾误用「拿破仑」「金枪鱼」）——_demo() 有断言对着语料答案集守着。
_PROMPT_EXAMPLE_ANSWER = "马目犬"  # 人造“答案”
_PROMPT_EXAMPLE_GUESS = "马猫目"  # 人造“猜测”，用来演示 🟩⬜🟨 三种标记
_PROMPT_EXAMPLE_WORDS: tuple[str, ...] = ("马目犬", "马猫目", "马目", "妈木")

W_LEGEND_TRI_ZH = """你会给出一个猜测，我会用逐位反馈告诉你猜得如何：
🟩 该位上的字正确，且位置正确
🟨 这个字在答案里出现，但位置不对
⬜ 答案中没有这个字
反馈示例（此处把答案设为人造词「马目犬」，与任何真实题目无关）：猜测「马猫目」→ 🟩⬜🟨，含义是第1位「马」正确；“猫”不在答案中；“目”在答案里但不在第3位。"""

W_LEGEND_BIN_ZH = """每轮你给出一个猜测，我会用逐位反馈告诉你猜得如何：
🟩 该位上的字正确，且位置正确
⬜ 其他情况"""

W_LEGEND_TRI_EN = """You give a guess, and I reply with per-position feedback:
🟩 this character is correct and in the correct position
🟨 this character occurs in the answer but in a wrong position
⬜ this character is not in the answer
Example (the answer is set to the artificial word 马目犬, unrelated to any real item):
guess 马猫目 → 🟩⬜🟨 — position 1 is correct; 猫 is not in the answer; 目 is in the answer but not at position 3."""

W_LEGEND_BIN_EN = """Each round you give one guess, and I reply with per-position feedback:
🟩 this character is correct and in the correct position
⬜ otherwise"""

W_STRATEGY_ZH = (
    """策略：先用反馈锁定确定的字，再逐步收敛；不要重复猜同一个词；每轮只输出一个猜测。"""
)

W_STRATEGY_EN = """Strategy: lock down confirmed characters first, then converge; never repeat a previous guess; output exactly one guess per round."""


@dataclass
class WordleConfig:
    """Wordle 式多轮口径（可做消融的变量集合）。"""

    max_turns: int = 6
    reveal_length: bool = True  # 是否告知答案字数
    feedback: str = "tri"  # tri | bin | phon | phon_char | phon_near（见模块 docstring）
    candidate_set: bool = False  # 是否给固定候选字池（按位候选的并集，不分组）
    reject_out_of_vocab: bool = True  # candidate_set=True 时，越界猜测判为非法、不计一轮
    output_mode: str = "guess"  # "guess" 只输出猜测 | "json" {"guess": "..."}
    resend_images: bool = True  # 每轮重发两图（无状态请求，推荐）
    # 答案的“冻结读音”，用于多音字消歧：[(声母, 韵母, 声调), ...]，长度=len(answer)。
    # 不给则用 pypinyin 的默认主读音（行的默认是 xing2；若答案是“银行”应显式传入 ("∅","in","2") + ("h","ang","2")）。
    gold_phon: Sequence[tuple[str, str, str]] | None = None

    def validate(self) -> WordleConfig:
        if self.feedback not in ("tri", "bin", *PHON_FEEDBACK_MODES):
            raise ValueError(f"feedback must be one of {('tri', 'bin', *PHON_FEEDBACK_MODES)}")
        if self.output_mode not in ("guess", "json"):
            raise ValueError("output_mode must be 'guess' or 'json'")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        return self


# --- 音系反馈：字 / 声母 / 韵母 / 声调（对齐中文 Wordle 变体「汉兜 Handle」的四属性反馈）---
# 设计文档：paper/11_评测提示词设计.md §Wordle 多轮

ATTR_CHAR, ATTR_INITIAL, ATTR_FINAL, ATTR_TONE = "char", "initial", "final", "tone"
ATTR_LABEL_ZH = {ATTR_CHAR: "字", ATTR_INITIAL: "声母", ATTR_FINAL: "韵母", ATTR_TONE: "声调"}
ATTR_LABEL_EN = {ATTR_CHAR: "char", ATTR_INITIAL: "initial", ATTR_FINAL: "final", ATTR_TONE: "tone"}

PHON_FEEDBACK_MODES = ("phon", "phon_char", "phon_near")
NA_MARK = "na"  # 该位不是汉字（字母/数字）→ 音系属性不适用，不参与比较

_ATTRS_PHON3: tuple[str, ...] = (ATTR_INITIAL, ATTR_FINAL, ATTR_TONE)
_ATTRS_PHON4: tuple[str, ...] = (ATTR_CHAR, *_ATTRS_PHON3)
_PHON_IDX = {ATTR_INITIAL: 0, ATTR_FINAL: 1, ATTR_TONE: 2}
_ATTR_SYM = {GREEN: "🟩", YELLOW: "🟨", GRAY: "⬜", NA_MARK: "·"}
ZERO_INITIAL = "∅"  # 零声母（鸭/压/鱼/安/五…）

# ---- 切分修正层（pypinyin 0.55 实测；见 paper/11 §5.2）----
# pypinyin 的 strict 切分对少数音节会给“合法但音系上错误”的结果（它的 _FINALS 表里
# 空韵母 '' 是合法成员，所以不会报错），必须先修正再参与属性比较。

# ① 舌尖元音被并进 'i'：zi/ci/si 的 [ɿ]、zhi/chi/shi/ri 的 [ʅ]、ji/qi/xi 的 [i] 全记 'i'。
#    phon_near 会合并 zh→z / ch→c / sh→s，若不区分韵母，「资(z,i) vs 知(zh,i)」会被判完全同音。
FINAL_I_FRONT = "iA"  # [ɿ] 舌尖前：资/词/思/自/此/死
FINAL_I_BACK = "iB"  # [ʅ] 舌尖后：知/吃/诗/日/支/事
_APICAL_FRONT = ("z", "c", "s")
_APICAL_BACK = ("zh", "ch", "sh", "r")

# ② 成音节叹词：韵母被记成空串、m/n 被当成声母（hm→h+''、m→m+''、ng→n+''、yo→''+o）。
#    按无调音节串整体覆盖。
_SYLLABLE_OVERRIDES = {
    "hm": ("h", "m"),
    "hng": ("h", "ng"),
    "m": (ZERO_INITIAL, "m"),
    "n": (ZERO_INITIAL, "n"),
    "ng": (ZERO_INITIAL, "ng"),
    "yo": (ZERO_INITIAL, "io"),
    "io": (ZERO_INITIAL, "io"),  # 注音 ㄧㄛ 的独立拼法（与 yo 同音；pypinyin 给空韵母）
}

# ③ 台湾读音 → 大陆普通话归一。pypinyin 只带大陆字典（contrib/ 里没有台湾表，
#    也没有 TW/ROC style），台湾标准保留了一些大陆已合流的读音，其音节不在 _FINALS 表里、
#    或组合在大陆非法：
#      ㄧㄞ (yai/iai) → ∅+ai   （丢介音，与「爱」同音）
#      ㄌㄩㄢ/ㄋㄩㄢ     → l/n+van（van 只允许跟在 j/q/x/∅ 后）
#    **完备性（已枚举）**：注音 21 聲母与 _INITIALS 完全相同；38 个韻母单位里只有 ㄧㄛ(io)
#    与 ㄧㄞ(iai) 是 _FINALS 表示不了的。所以「台湾能写、大陆拼音写不出」的类就这些，已穷尽。
#    键 = 无调音节串（TONE3 去掉末位数字），值 = 大陆对应的 (声母, 韵母)。
_TAIWAN_OVERRIDES = {
    "yai": (ZERO_INITIAL, "ia"),  # 崖/涯/睚：台 ㄧㄞˊ → 陆 yá
    "iai": (ZERO_INITIAL, "ia"),  # 同上，ㄧㄞ 的另一种拼法
    "lvan": ("l", "uan"),  # 攣/孌/圞：台 ㄌㄩㄢˊ → 陆 luán
    "nvan": ("n", "uan"),  # l/n 对称：l、n 后都不允许 üan
}

# ④ 韵母级兜底：ü 的拼写形式（'lüan'/'nüan'）或别的 pypinyin 版本可能绕过 ③ 的音节表，
#    直接递进 l/n + 'van' → 一律折回 'uan'（l/n 后只允许 've'：略 lüe / 虐 nüe，不受影响）
_FINAL_NORM_BY_INITIAL = {("l", "van"): "uan", ("n", "van"): "uan"}

# 近音合并表（仅 feedback="phon_near" 生效；口径对齐 pun_scan 的宽匹配）
NEAR_INITIAL = {"zh": "z", "ch": "c", "sh": "s", "n": "l", "r": "l", "f": "h"}
NEAR_FINAL = {"ing": "in", "eng": "en", "iang": "ian", "uang": "uan", "ang": "an", "ong": "on"}

# 反馈格式的示意例子（与任何具体题目无关，避免泄漏）：答案设为人造「马目」，猜测「妈木」
_PHON_EX_GUESS = ("妈", "木")
_PHON_EX_MARKS = {
    ATTR_CHAR: [GRAY, GRAY],
    ATTR_INITIAL: [GREEN, GREEN],
    ATTR_FINAL: [GREEN, GREEN],
    ATTR_TONE: [GRAY, GREEN],
}


def attrs_for(feedback: str) -> tuple[str, ...]:
    """反馈档 → 参与反馈的属性集合。"""
    if feedback in ("tri", "bin"):
        return (ATTR_CHAR,)
    if feedback == "phon":
        return _ATTRS_PHON3
    if feedback in ("phon_char", "phon_near"):
        return _ATTRS_PHON4
    raise ValueError(f"unknown feedback mode: {feedback!r}")


def _pypinyin() -> Any:
    """延迟导入 pypinyin（未安装时返回 None）。

    注意：音系档（phon*）**必须**有 pypinyin，缺失时由 require_pypinyin() 显式报错，
    不会静默退化成“全灰”——那会在评测里制造无声的错误结果。
    """
    try:
        import pypinyin

        return pypinyin
    except Exception:  # noqa: BLE001 - 把「没装 pypinyin」与「装了但导入炸了」一同当缺失处理
        return None


def require_pypinyin() -> Any:
    """取 pypinyin，缺失则报错并给出安装提示。"""
    pinyin = _pypinyin()
    if pinyin is None:
        raise RuntimeError(
            "音系反馈（feedback='phon'/'phon_char'/'phon_near'）依赖 pypinyin，但当前解释器里没有装。"
            "请先 `pip install pypinyin`（注意：本仓库 .venv 目前未安装，只有基底解释器里有）。"
        )
    return pinyin


def _norm_syllable(initial: str, final: str, tone3: str) -> tuple[str, str, str] | None:
    """把 pypinyin 的 (声母, 韵母, 带调音节) 修正成我们的音系三元组；无法确定时返回 None。

    tone3 形如 'zhong1' / 'zi' / 'hng'（轻声与成音节叹词没有数字）。
    修正项对应文件头部的 ①②③④ 四张表，**勿随意改**，改动需同步 paper/11 §5.2。
    """
    toneless = tone3[:-1] if tone3 and tone3[-1].isdigit() else tone3
    if toneless in _SYLLABLE_OVERRIDES:  # ② 成音节叹词 / 注音 ㄧㄛ
        initial, final = _SYLLABLE_OVERRIDES[toneless]
    elif toneless in _TAIWAN_OVERRIDES:  # ③ 台湾读音
        initial, final = _TAIWAN_OVERRIDES[toneless]
    else:
        if final == "i":  # ① 舌尖元音
            if initial in _APICAL_FRONT:
                final = FINAL_I_FRONT
            elif initial in _APICAL_BACK:
                final = FINAL_I_BACK
        final = _FINAL_NORM_BY_INITIAL.get((initial, final), final)  # ④ 非法组合
        if not final:
            # 韵母为空 = 这个拼法 pypinyin 根本表示不了（通用拼音 zih/jhih/chih… 的 -ih、vun）。
            # **绝不能静默留一个空串**——空串之间会互相判等，在评测里制造假同音。
            # 返回 None → 该读音被丢弃，该位音系属性记 NA（可见、不参与比较）。
            return None
    tone = tone3[-1] if tone3 and tone3[-1].isdigit() else "5"
    return initial, final, tone


@lru_cache(maxsize=16384)
def phon_profile(ch: str) -> tuple[tuple[str, str, str], ...] | None:
    """汉字 → 全部读音三元组 (声母, 韵母, 声调)；**第一项是主读音**；非汉字 / 无 pypinyin → None。

    口径（勿随意改，改动需同步 paper/11）：
    - 声母取 Style.INITIALS(strict=True)，零声母记 '∅'（鸭/压/鱼/安/五…）
    - 韵母取 Style.FINALS(strict=True)，ü 记作 'v'（绿 lv4 ≠ 路 lu4，**不能**把 v 归一成 u）
    - 声调取 Style.TONE3(strict=True) 的末位数字（轻声无数字 → 记 5）
    - 上面三值再过一遍 _norm_syllable()：舌尖元音 iA/iB、成音节叹词 hm/hng/m/n/ng/yo/io、
      台湾读音（yai/iai/lvan/nvan）、l/n+van→uan（四张修正表见文件头部）；
      修正后韵母仍为空 = pypinyin 表示不了的拼法（如通用拼音 zih/jhih）→ 整字返回 None

    多音字的正确取法（实测 pypinyin 0.55，两条错路都别走）：
    - ❌ 把各 style 的 heteronym 列表按下标配对：三个列表**各自去重且长度不同**
      （行 = INITIALS 2 项 / FINALS 3 项 / TONE3 5 项），配出来会有 'x'+'ang' 这种不存在的音节。
    - ❌ 直接切音节字符串：'yā' 应拆成 ∅+ia、'lǜ' 应拆成 l+v，按“首字母=声母”切会错。
    - ✅ 用 pypinyin 自己的读音表 pinyin_dict（每字全部读音）配合 style.convert(..., strict=True)
      逐读音转换——它天然处理零声母与 ü。
    - 若 pinyin_dict/convert 不可用（别的 pypinyin 版本）→ 只返回主读音，宁可保守也不造假。
    """
    pinyin = _pypinyin()
    if pinyin is None or len(ch) != 1 or not ("\u3400" <= ch <= "\u9fff"):
        return None

    try:
        primary = _norm_syllable(
            pinyin.pinyin(ch, style=pinyin.Style.INITIALS, strict=True)[0][0] or ZERO_INITIAL,
            pinyin.pinyin(ch, style=pinyin.Style.FINALS, strict=True)[0][0],
            pinyin.pinyin(ch, style=pinyin.Style.TONE3, strict=True)[0][0],
        )
    except Exception:  # noqa: BLE001 - 取不到主读音就整字当不可用（音系属性记 NA），不猜
        return None
    if primary is None:  # 主读音都表示不了 → 整字当不可用（音系属性记 NA）
        return None

    triples: list[tuple[str, str, str]] = []
    try:
        from pypinyin.pinyin_dict import pinyin_dict
        from pypinyin.style import convert

        for one in pinyin_dict[ord(ch)].split(","):
            tri = _norm_syllable(
                convert(one, pinyin.Style.INITIALS, True) or ZERO_INITIAL,
                convert(one, pinyin.Style.FINALS, True),
                convert(one, pinyin.Style.TONE3, True),
            )
            if tri is not None:  # 表示不了的读音直接丢弃，不参与属性比较
                triples.append(tri)
    except Exception:  # noqa: BLE001 - 换 pypinyin 版本时 pinyin_dict/convert 可能没有；
        triples = []  # 此时只保留主读音，宁可保守也不造假

    out: list[tuple[str, str, str]] = []
    for tri in [primary, *triples]:  # 主读音永远排第一，且去重
        if tri not in out:
            out.append(tri)
    return tuple(out)


@lru_cache(maxsize=16384)
def is_polyphone(ch: str) -> bool:
    """该字是否有多个读音（多音字 / 轻声歧义）→ 它在答案里的实际读音需要冻结（gold_phon）。"""
    prof = phon_profile(ch)
    return prof is not None and len(prof) > 1


def suggest_gold_phon(text: str) -> tuple[list[tuple[str, str, str]], list[tuple[int, str]]]:
    """给一条答案生成建议的 gold_phon，并列出**需要人工确认**的多音字位置。

    返回 (triples, review)：
      triples —— 逐字的主读音三元组，可直接塞进 WordleConfig(gold_phon=triples)
      review  —— [(位置(1-based), 字), ...]，这些位置在上下文里可能是另一个读音
                 （如「重庆」的重应为 chong2，而主读音给的是 zhong4），必须冻结而非放默认。

    建议：正式跑评测前对全部 10k 条做一次 suggest_gold_phon，把 review 过一遍并用 LLM/人工
    按答案词义冻结读音；之后所有实验都用冻结值，保证可复现且不受 pypinyin 版本影响。
    """
    triples: list[tuple[str, str, str]] = []
    review: list[tuple[int, str]] = []
    for i, ch in enumerate(text, 1):
        prof = phon_profile(ch)
        if prof is None:
            triples.append((ZERO_INITIAL, ch, "5"))  # 非汉字：占位，音系属性会判 NA
            continue
        triples.append(prof[0])
        if is_polyphone(ch):
            review.append((i, ch))
    return triples, review


def _same_attr(attr: str, x: str, y: str, near: bool) -> bool:
    if not near:
        return x == y
    if attr == ATTR_INITIAL:
        return NEAR_INITIAL.get(x, x) == NEAR_INITIAL.get(y, y)
    if attr == ATTR_FINAL:
        return NEAR_FINAL.get(x, x) == NEAR_FINAL.get(y, y)
    return x == y


def _gold_attr_values(
    attr: str, gold: str, gold_phon: Sequence[tuple[str, str, str]] | None
) -> list[str | None]:
    vals: list[str | None] = []
    for i, ch in enumerate(gold):
        if attr == ATTR_CHAR:
            vals.append(ch)
        elif gold_phon and i < len(gold_phon):
            vals.append(gold_phon[i][_PHON_IDX[attr]])
        else:
            prof = phon_profile(ch)
            vals.append(prof[0][_PHON_IDX[attr]] if prof else None)
    return vals


def _guess_attr_variants(attr: str, guess: str) -> list[list[str]]:
    out: list[list[str]] = []
    for ch in guess:
        if attr == ATTR_CHAR:
            out.append([ch])
        else:
            prof = phon_profile(ch)
            out.append([v[_PHON_IDX[attr]] for v in prof] if prof else [])
    return out


def phon_marks(
    guess: str,
    gold: str,
    cfg: WordleConfig,
    gold_phon: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, list[str]]:
    """逐属性逐位判定。**每项属性独立**跑一遍两遍扫（先锁定位对，再从池里扣减），
    否则重复音节会重复计数，且字属性与音系属性会互相污染。"""
    require_pypinyin()  # 缺 pypinyin 直接报错，不接受静默降级
    attrs = attrs_for(cfg.feedback)
    near = cfg.feedback == "phon_near"
    gold_phon = gold_phon if gold_phon is not None else cfg.gold_phon
    n = len(guess)
    gold_vals = {t: _gold_attr_values(t, gold, gold_phon) for t in attrs}
    guess_vars = {t: _guess_attr_variants(t, guess) for t in attrs}

    result: dict[str, list[str]] = {}
    for t in attrs:
        marks = [GRAY] * n
        pool: dict[str, int] = {}
        for i in range(n):
            if i >= len(gold_vals[t]):  # 猜得比答案长：该位答案里不存在
                continue
            av = gold_vals[t][i]
            if av is None:  # 该位不是汉字 → 音系属性不适用
                marks[i] = NA_MARK
                continue
            if any(_same_attr(t, x, av, near) for x in guess_vars[t][i]):
                marks[i] = GREEN
            else:
                pool[av] = pool.get(av, 0) + 1
        for i in range(n):
            if marks[i] != GRAY:
                continue
            for x in guess_vars[t][i]:
                hit = next(
                    (k for k, c in pool.items() if c > 0 and _same_attr(t, x, k, near)), None
                )
                if hit is not None:
                    marks[i] = YELLOW
                    pool[hit] -= 1
                    break
        result[t] = marks
    return result


def render_phon_feedback(
    guess: str,
    gold: str,
    cfg: WordleConfig,
    lang: str = "zh",
    gold_phon: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """渲染多属性反馈：第一行是猜测的字（逐位对齐），下面每行一项属性。"""
    attrs = attrs_for(cfg.feedback)
    marks = phon_marks(guess, gold, cfg, gold_phon)
    labels = ATTR_LABEL_ZH if lang == "zh" else ATTR_LABEL_EN
    pad = 3 if lang == "zh" else 8

    def _row(label: str, cells: Sequence[str]) -> str:
        return f"{label}{' ' * (pad - len(label))}" + " ".join(cells)

    grid = "\n".join(
        [_row("", list(guess))] + [_row(labels[t], [_ATTR_SYM[m] for m in marks[t]]) for t in attrs]
    )
    detail = []
    for i, ch in enumerate(guess, 1):
        cells = "｜".join(f"{labels[t]}{_ATTR_SYM[marks[t][i - 1]]}" for t in attrs)
        detail.append(f"第{i}位「{ch}」：{cells}" if lang == "zh" else f"pos {i} '{ch}': {cells}")
    head = f"猜测「{guess}」" if lang == "zh" else f'guess "{guess}"'
    return {
        "marks": marks,
        "grid": grid,
        "detail": detail,
        "rendered": f"{head}\n{grid}",
        "multiline": True,
    }


def _phon_legend(cfg: WordleConfig, lang: str = "zh") -> str:
    """音系档的规则说明（含示意例子；例子故意用人造配对，不泄漏本题答案）。"""
    attrs = attrs_for(cfg.feedback)
    zh = lang == "zh"
    labels = ATTR_LABEL_ZH if zh else ATTR_LABEL_EN
    pad = 3 if zh else 8
    rows = [" " * pad + " ".join(_PHON_EX_GUESS)]
    rows += [
        f"{labels[t]}{' ' * (pad - len(labels[t]))}"
        + " ".join(_ATTR_SYM[m] for m in _PHON_EX_MARKS[t])
        for t in attrs
    ]
    example = "\n".join(rows)

    if zh:
        return "\n".join(
            [
                "你会给出一个猜测，我会给出「属性 × 位置」的逐位反馈。属性是："
                + "、".join(labels[t] for t in attrs)
                + "。",
                "符号含义：",
                "🟩 该属性在这一位与答案一致",
                "🟨 该属性的值出现在答案的其他位置，但不在这一位",
                "⬜ 答案中没有该属性的值",
                "·  这一位不是汉字（字母/数字），该属性不适用",
                f"反馈格式（第1行是你猜的字，逐位对齐；下面每行一项属性）：\n{example}",
                "上例是示意（与本题无关）：你猜的两个字汉字都不对，但读音都对——"
                "第1位「妈 mā」与答案该位声母、韵母相同而声调不同；"
                "第2位「木 mù」与答案该位完全同音，只是写成了另一个字。",
                "本题考的是“读音相同或相近”，所以三项音系属性都一致、只是汉字不同，说明读音很可能已经对了。",
                "",
                "本档的取值约定（与常见写法不同，否则“属性一致”会被理解错）：",
                "- 按普通话、逐字读音拆，不做变调（一/不）与儿化。",
                "- 声母：零声母记 `∅`（鸭/压/安/鱼/五 的声母是 `∅`，不是 y/w）。",
                "- 韵母：`z/c/s` 后的 i 记 `iA`（资/词/思），`zh/ch/sh/r` 后的 i 记 `iB`（知/吃/诗/日），"
                "其余 i 仍是 `i`（机/七/西）——`iA`、`iB`、`i` 是三个不同的韵母属性；"
                "记 `iA` 和 `iB` 是因为这两类 i 与 `i` 在普通话里读音本来就不同（舌尖元音）。",
                "- 韵母的 ü 一律记 `v`：绿 `lv` 与 路 `lu` 不同；另外 `ui/iu/un` 按 `uei/iou/uen` 写，"
                "成音节叹词记 `m`/`n`/`ng`，`ê`（欷）另作一个韵母。",
                "- 声调用数字 `1/2/3/4/5`，其中 `5` 是轻声（子/的/们），不是“第五声”。",
                "- 猜测比答案长的位一律 ⬜；非汉字位（字母/数字）音系属性记 `·`。",
            ]
            + (
                [
                    "",
                    "本档另有近音合并（只在本档生效）：声母 zh→z、ch→c、sh→s、n→l、r→l、f→h；"
                    "韵母 ing→in、eng→en、iang→ian、uang→uan、ang→an、ong→on。",
                    "合并发生在属性判定时：所以「早 z+ao」与「找 zh+ao」在本档下声母判 🟩（读音视为相近，不是判错）。",
                    "但舌尖元音不参与合并：「资 z+iA」与「知 zh+iB」即使合并了 zh→z，声母 🟩 而韵母仍 ⬜（`iA ≠ iB`）。",
                ]
                if cfg.feedback == "phon_near"
                else []
            )
        )

    return "\n".join(
        [
            "You give a guess; I reply with per-attribute, per-position feedback. Attributes: "
            + ", ".join(labels[t] for t in attrs)
            + ".",
            "Symbols:",
            "🟩 this attribute matches the answer at this position",
            "🟨 this attribute's value occurs elsewhere in the answer, but not at this position",
            "⬜ the answer does not contain this attribute's value",
            "·  this position is not a Han character (letter/digit), so the attribute does not apply",
            f"Feedback format (row 1 is your guess, aligned by position):\n{example}",
            "The example above is illustrative and unrelated to this puzzle: both guessed characters are the wrong "
            "characters but their readings are right - position 1 shares initial and final with the answer and differs "
            "only in tone; position 2 is a perfect homophone written with another character.",
            "",
            "How attribute values are written here (differs from the usual notation):",
            "- Standard Mandarin, character by character; no tone sandhi (一/不) and no erhua.",
            "- Initial: a zero initial is written `∅` (duck/安/鱼/五 have `∅`, not y/w).",
            "- Final: the i after z/c/s is written `iA` (资/词/思), the i after zh/ch/sh/r is written `iB` "
            "(知/吃/诗/日), any other i stays `i` (机/七/西) - `iA`, `iB` and `i` are THREE different finals; "
            "the two apical vowels really are distinct from `i` in Mandarin.",
            "- Final u-umlaut is always written `v`: green `lv` differs from road `lu`. Also ui/iu/un are written "
            "uei/iou/uen, syllabic interjections as `m`/`n`/`ng`, and `ê` has its own final.",
            "- Tone is a digit 1/2/3/4/5 where 5 is the neutral tone (子/的/们), not a fifth tone.",
            "- Positions beyond the answer's length are always ⬜; non-Han positions (letters/digits) give `·`.",
        ]
        + (
            [
                "",
                "Near-sound merging (this tier only): initials zh→z, ch→c, sh→s, n→l, r→l, f→h; "
                "finals ing→in, eng→en, iang→ian, uang→uan, ang→an, ong→on.",
                "Merging happens at ATTRIBUTE comparison time, so 早 z+ao and 找 zh+ao score 🟩 on the initial here "
                "(their readings count as close, it is not a mistake).",
                "The apical vowels do NOT merge: 资 z+iA vs 知 zh+iB keeps ⬜ on the final even after zh→z collapses "
                "the initial (`iA ≠ iB`).",
            ]
            if cfg.feedback == "phon_near"
            else []
        )
    )


def wordle_marks(guess: str, gold: str) -> list[str]:
    """Wordle 三态逐位判定（含重复字只标一次）。返回长度 == len(guess)。"""
    n = len(guess)
    marks = [GRAY] * n
    pool: dict[str, int] = {}
    for i, ch in enumerate(guess):
        if i < len(gold) and ch == gold[i]:
            marks[i] = GREEN
        elif i < len(gold):
            pool[gold[i]] = pool.get(gold[i], 0) + 1
    for i, ch in enumerate(guess):
        if marks[i] == GREEN:
            continue
        if pool.get(ch, 0) > 0:
            marks[i] = YELLOW
            pool[ch] -= 1
    return marks


def render_feedback(
    guess: str,
    gold: str,
    cfg: WordleConfig,
    lang: str = "zh",
    gold_phon: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """生成一轮反馈：{marks, grid, detail, rendered, multiline}。

    feedback 为音系档（phon*）时委派给 render_phon_feedback，gold_phon 一并透传
    （多音字答案必须先冻结读音，否则声母/韵母/声调三项属性会按错读音算）；
    否则是字级反馈，长度不一致的处理口径：按较短者逐位对齐，猜测超出的位置一律 ⬜。
    """
    if cfg.feedback in PHON_FEEDBACK_MODES:
        return render_phon_feedback(guess, gold, cfg, lang=lang, gold_phon=gold_phon)
    marks = wordle_marks(guess, gold)
    if cfg.feedback == "bin":
        marks = [GREEN if m == GREEN else GRAY for m in marks]
    grid = "".join(GRID_CHAR[m] for m in marks)
    detail: list[str] = []
    for i, (ch, m) in enumerate(zip(guess, marks, strict=False), 1):
        if m == GREEN:
            s = "位置正确"
        elif m == YELLOW:
            s = "字在答案中但位置不对"
        else:
            s = "答案中没有这个字"
        detail.append(f"第{i}位“{ch}”：{s}" if lang == "zh" else f"position {i} '{ch}': {s}")
    rendered = (
        f"猜测「{guess}」→ {grid}（{'；'.join(detail)}）"
        if lang == "zh"
        else f'guess "{guess}" → {grid} ({"; ".join(detail)})'
    )
    return {
        "marks": marks,
        "grid": grid,
        "detail": detail,
        "rendered": rendered,
        "multiline": False,
    }


def _wordle_system(
    cfg: WordleConfig, lang: str = "zh", length: int = 3, pool: Sequence[str] | None = None
) -> str:
    zh = lang == "zh"
    parts: list[str] = []
    parts.append(
        "你正在玩一个中文谐音猜谜游戏（Wordle 式）。"
        if zh
        else "You are playing a Wordle-style Chinese homophone guessing game."
    )
    if cfg.reveal_length:
        parts.append(
            (
                f"谜题给出两张卡片图片（图1提示卡印“这是X”，图2猜测卡提示“这是___”），"
                f"答案有 {length} 个字；你至多有 {cfg.max_turns} 次猜测机会。"
            )
            if zh
            else (
                f'Two card images are given (card 1 hint card prints "this is X"; '
                f'card 2 prints "this is ___"). The answer has {length} characters; '
                f"you have at most {cfg.max_turns} guesses."
            )
        )
    else:
        parts.append(
            (
                f"谜题给出两张卡片图片（图1提示卡印“这是X”，图2猜测卡提示“这是___”）；"
                f"答案的字数不告诉你，需要你自己判断；你至多有 {cfg.max_turns} 次猜测机会。"
            )
            if zh
            else (
                f'Two card images are given (card 1 hint card prints "this is X"; '
                f'card 2 prints "this is ___"). The length of the answer is NOT given; '
                f"you have at most {cfg.max_turns} guesses."
            )
        )
    if cfg.feedback in PHON_FEEDBACK_MODES:
        parts.append(_phon_legend(cfg, lang=lang))
    else:
        parts.append(
            W_LEGEND_TRI_ZH
            if cfg.feedback == "tri" and zh
            else W_LEGEND_TRI_EN
            if cfg.feedback == "tri"
            else W_LEGEND_BIN_ZH
            if zh
            else W_LEGEND_BIN_EN
        )
    if cfg.candidate_set and pool:
        head = (
            "可用字池（只能从中取字，可重复使用）："
            if zh
            else "Available character pool (draw only from these):"
        )
        parts.append(head + " " + " ".join(pool))
    parts.append(W_STRATEGY_ZH if zh else W_STRATEGY_EN)
    if cfg.feedback in PHON_FEEDBACK_MODES:
        parts.append(
            "本档的用法：先用三项音系属性把每一“位”的读音定下来，再决定用哪个字；"
            "某个字属性是 ⬜ 而声母/韵母/声调都是 🟩 时，那是同音异字，不是错。"
            if zh
            else "How to use this mode: first pin down the reading of each position via the three phonological "
            "attributes, then decide the character. A ⬜ on the char attribute with 🟩 on initial/final/tone means a "
            "homophone written with a different character - not an error."
        )
    if cfg.output_mode == "json":
        parts.append(
            '输出要求：只输出一个 JSON 对象：{"guess": "你的猜测"}，不要任何多余文字。'
            if zh
            else 'Output requirements: output exactly one JSON object: {"guess": "..."} and nothing else.'
        )
    else:
        parts.append(
            (
                "输出要求：只输出你的猜测本身，不要输出解释、拼音、序号或代码块。"
                if cfg.reveal_length
                else "输出要求：只输出你的猜测本身（长度自行判断），不要输出解释、拼音、序号或代码块。"
            )
            if zh
            else (
                "Output requirements: output only your guess, no explanation, pinyin, numbering or code fences."
                if cfg.reveal_length
                else "Output requirements: output only your guess (decide the length yourself), no explanation or fences."
            )
        )
    return "\n\n".join(parts)


def new_wordle_state(item: dict[str, Any]) -> dict[str, Any]:
    """初始化一题的 Wordle 会话状态。"""
    return {
        "id": item.get("id"),
        "gold": item["answer"],
        "history": [],
        "invalid": [],
        "turns_used": 0,
    }


def build_wordle_turn(
    item: dict[str, Any],
    history: Sequence[str],
    cfg: WordleConfig | None = None,
    turn: int | None = None,
    turn_notes: Sequence[str] = (),
    lang: str = "zh",
    image_encoder: Callable[[str], str] = encode_image_data_url,
) -> list[dict[str, Any]]:
    """构造第 turn 轮的 messages（无状态：重发两图 + 文本历史 + 本轮要求）。

    history   : 历史猜测字符串列表（顺序即轮次）
    turn_notes: 与 history 对齐的额外提示（如“你的猜测为2个字，答案为3个字”），可空
    """
    cfg = (cfg or WordleConfig()).validate()
    zh = lang == "zh"
    gold = item["answer"]
    turn = turn or (len(history) + 1)

    pool: list[str] | None = None
    if cfg.candidate_set:
        seen: list[str] = []
        for pos in item.get("positions", []):
            for c in pos.get("candidates", []):
                if c not in seen:
                    seen.append(c)
        pool = seen

    lines: list[str] = []
    if history:
        lines.append("已猜记录（按轮次）：" if zh else "Guess history:")
        for i, g in enumerate(history, 1):
            fb = render_feedback(g, gold, cfg, lang=lang)
            note = turn_notes[i - 1] if turn_notes and i - 1 < len(turn_notes) else ""
            block = fb["rendered"]
            if note:
                block = f"{block}\n{note}" if fb.get("multiline") else f"{block}（{note}）"
            lines.append((f"第{i}轮 " if zh else f"round {i} ") + block)
        lines.append("")
    if zh:
        lines.append(f"请给出第 {turn} 轮猜测（第 {turn}/{cfg.max_turns} 次机会）。")
    else:
        lines.append(f"Give guess #{turn} (attempt {turn}/{cfg.max_turns}).")
    ask = "\n".join(lines)

    urls = []
    if cfg.resend_images and item.get("hint_img") and item.get("guess_img"):
        urls = [image_encoder(item["hint_img"]), image_encoder(item["guess_img"])]
        head = (
            "第一张图 = 图1（提示卡），第二张图 = 图2（猜测卡）。\n"
            if zh
            else "The first image is card 1 (hint card); the second is card 2 (guess card).\n"
        )
        ask = head + ask

    return [
        {"role": "system", "content": _wordle_system(cfg, lang=lang, length=len(gold), pool=pool)},
        {"role": "user", "content": _user_content(ask, urls)},
    ]


def wordle_length_note(guess: str, gold: str, cfg: WordleConfig, lang: str = "zh") -> str:
    """reveal_length=True 且长度不符时，附加的提示文本（可放进 turn_notes）。"""
    if not cfg.reveal_length or len(guess) == len(gold):
        return ""
    if lang == "zh":
        return f"注意：你的猜测为 {len(guess)} 个字，答案是 {len(gold)} 个字。"
    return f"Note: your guess has {len(guess)} characters; the answer has {len(gold)}."


# ============================================================================
# 4. 解析（与提示词契约配套）
# ============================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _loads_lenient(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    for cand in [text, *_FENCE_RE.findall(text)]:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001 - 解析失败就试下一种候选，这里就是要容错
            pass
        i, j = cand.find("{"), cand.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(cand[i : j + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001 - 同上：大括号截取只是最后一搏
                pass
    return None


def parse_t1(text: str, output_mode: str = "json") -> dict[str, Any]:
    """解析 T1 输出。json 档失败时回退到“取最后一个非空行”的答案抽取。"""
    if output_mode == "json":
        obj = _loads_lenient(text)
        if obj:
            ans = str(obj.get("answer", "") or "").strip()
            if not ans:
                m = re.search(r'"answer"\s*:\s*"([^"]*)"', text or "")
                ans = m.group(1).strip() if m else ""
            return {
                "format_ok": bool(ans),
                "answer": ans,
                "raw_json": obj,
                **{
                    k: obj.get(k)
                    for k in ("hint_name", "guess_name", "hint_pinyin", "guess_pinyin")
                },
            }
        return {"format_ok": False, "answer": "", "raw_json": None}
    return {"answer": parse_answer_only(text), "format_ok": bool(parse_answer_only(text))}


_PUNCT = " \t\r\n。，、；：！？!?.,;:\"'“”‘’（）()【】[]{}<>《》…·-—_*#`"


def parse_answer_only(text: str, max_len: int = 12) -> str:
    """从自由文本里抽取答案：优先最后一行去标点/序号/引号，超长再退化为整串。"""
    raw = (text or "").strip()
    for line in reversed([ln for ln in raw.splitlines() if ln.strip()]):
        s = line.strip().strip("`").strip()
        s = re.sub(r"^[\s\d]+[.、)．]\s*", "", s)
        s = s.strip(_PUNCT)
        if 0 < len(s) <= max_len:
            return s
    return raw.strip(_PUNCT)


def parse_t4(text: str, output_mode: str = "letters") -> dict[str, Any]:
    """解析 T4 输出。letters 档取答案串；json 档取 choices 拼接并给逐位结果。"""
    if output_mode == "json":
        obj = _loads_lenient(text)
        if obj and isinstance(obj.get("choices"), list):
            ch = [str(c).strip() for c in obj["choices"]]
            return {"format_ok": all(len(c) == 1 for c in ch), "choices": ch, "answer": "".join(ch)}
        return {"format_ok": False, "choices": [], "answer": ""}
    return {"answer": parse_answer_only(text), "format_ok": bool(parse_answer_only(text))}


def parse_wordle_guess(text: str, output_mode: str = "guess") -> dict[str, Any]:
    """解析一轮 Wordle 猜测。"""
    if output_mode == "json":
        obj = _loads_lenient(text)
        if obj and str(obj.get("guess", "")).strip():
            return {"format_ok": True, "guess": str(obj["guess"]).strip()}
        return {"format_ok": False, "guess": ""}
    g = parse_answer_only(text, max_len=10)
    return {"format_ok": bool(g), "guess": g}


# ============================================================================
# 5. 请求参数建议（供 harness 直接采用）
# ============================================================================


def chat_params(
    task: str = "t1",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """推荐的采样参数。主表用 greedy（temperature=0）；重复实验再放开。"""
    if max_tokens is None:
        max_tokens = {"t1": 256, "t4": 64, "wordle": 32}.get(task, 256)
    p: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        p["seed"] = seed
    return p


# ============================================================================
# 6. 自检 / 演示
# ============================================================================


def _first_text(content: Any) -> str:
    """从 messages 的 content 里取出文本部分（多模态 content 中可能不在第 0 位）。"""
    if isinstance(content, str):
        return content
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", ""))
    return ""


def _demo() -> None:
    with contextlib.suppress(Exception):  # Windows 控制台默认 GBK，避免中文打印报错
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    print(f"PROMPT_VERSION = {PROMPT_VERSION}\n")

    # --- 防泄漏自检：提示词里举例用的词绝不能是语料里的真实答案 ---
    # 语料路径优先取环境变量 PUNQUIZCN_CORPUS，否则尝试仓库内 data/ 下的两个常见位置。
    # 语料不随代码发布（体积原因），找不到时跳过检查而不报错。
    _corpus = corpus_path()
    if _corpus is not None:
        with _corpus.open(encoding="utf-8-sig", newline="") as _fh:
            _answers = {
                (r.get("answer") or "").strip() for r in csv.DictReader(_fh, delimiter="\t")
            }
        _answers.discard("")
        _leak = sorted(w for w in _PROMPT_EXAMPLE_WORDS if w in _answers)
        assert not _leak, f"提示词示例词出现在语料答案里（泄漏）：{_leak}"
        print(
            f"prompt-example leak check: OK "
            f"({len(_PROMPT_EXAMPLE_WORDS)} 个示例词 vs {len(_answers)} 个答案)\n"
        )
    else:
        print("语料文件不在，跳过示例词泄漏检查\n")

    # --- Wordle 反馈自检（含重复字） ---
    assert wordle_marks("金枪鱼", "金枪鱼") == [GREEN, GREEN, GREEN]
    assert wordle_marks("枪金鱼", "金枪鱼") == [YELLOW, YELLOW, GREEN]
    assert wordle_marks("金银铜", "金枪鱼") == [GREEN, GRAY, GRAY]
    assert wordle_marks("金金金", "金枪鱼") == [GREEN, GRAY, GRAY]  # 重复字只标一次
    assert wordle_marks("金银", "金枪鱼") == [GREEN, GRAY]  # 长度不足
    assert wordle_marks("金银铜铁", "金枪鱼") == [GREEN, GRAY, GRAY, GRAY]  # 长度超出
    assert wordle_marks("铁枪金", "金枪鱼") == [GRAY, GREEN, YELLOW]  # 位对 + 错位字
    print("wordle_marks self-test: OK\n")

    # --- 音系反馈自检（声母/韵母/声调） ---
    if _pypinyin() is None:
        print("pypinyin 未安装：跳过音系反馈自检（字级反馈不受影响）\n")
    else:
        cfg_pc = WordleConfig(feedback="phon_char")

        def prof(ch: str) -> tuple[tuple[str, str, str], ...]:
            """自检辅助：断言该字有读音（phon_profile 返回 Optional）。"""
            p = phon_profile(ch)
            assert p is not None, ch
            return p

        # 零声母统一为 ∅；ü 记 v
        assert prof("鸭")[0][:2] == prof("压")[0][:2] == (ZERO_INITIAL, "ia")
        assert prof("绿")[0][1] == "v" and prof("路")[0][1] == "u"  # 绿 lv4 ≠ 路 lu4
        assert prof("鱼")[0] == (ZERO_INITIAL, "v", "2")  # 零声母 + ü + 声调单独取
        assert prof("元")[0] == (ZERO_INITIAL, "van", "2")  # ü 家族统一用 ASCII 写法 (yuan)
        assert prof("弯")[0] == (ZERO_INITIAL, "uan", "1")  # 元 van ≠ 弯 uan，不许折成同一个
        # ① 舌尖元音：zi / zhi / ji 的韵母必须分开（pypinyin 一律给 'i'）
        assert prof("资")[0] == ("z", FINAL_I_FRONT, "1")
        assert prof("知")[0] == ("zh", FINAL_I_BACK, "1")
        assert prof("思")[0][1] == FINAL_I_FRONT and prof("诗")[0][1] == FINAL_I_BACK
        assert prof("日")[0][1] == FINAL_I_BACK and prof("机")[0][1] == "i"  # 舌面 [i] 不受影响
        # ②③④ 修正表（对 (声母, 韵母, 带调音节) 直接验证）
        assert _norm_syllable("h", "", "hm") == ("h", "m", "5")
        assert _norm_syllable("h", "", "hng2") == ("h", "ng", "2")
        assert _norm_syllable("m", "", "m") == (ZERO_INITIAL, "m", "5")
        assert _norm_syllable("n", "", "ng") == (ZERO_INITIAL, "ng", "5")
        assert _norm_syllable("", "o", "yo1") == (ZERO_INITIAL, "io", "1")
        assert _norm_syllable("", "ai", "yai2") == (ZERO_INITIAL, "ia", "2")  # 台 崖 yai → 陆 ya
        assert _norm_syllable("l", "van", "lvan2") == ("l", "uan", "2")  # 台 攣 lvan → 陆 luan
        assert _norm_syllable("n", "van", "nvan3") == ("n", "uan", "3")
        assert _norm_syllable("j", "van", "jvan1") == ("j", "van", "1")  # j 后的 üan 合法，不动
        assert _norm_syllable("", "", "io") == (ZERO_INITIAL, "io", "5")  # 注音 ㄧㄛ
        assert _norm_syllable("", "ai", "iai2") == (ZERO_INITIAL, "ia", "2")  # ㄧㄞ
        # 表示不了的拼法（通用拼音的 -ih / vun）→ None，绝不静默塞空韵母
        assert _norm_syllable("z", "", "zih") is None
        assert _norm_syllable("j", "", "jhih") is None
        assert _norm_syllable("", "", "vun") is None
        # ê（ㄝ，欸/誒）**在 _FINALS 里**，四张表都不碰它（零声母由调用方补 ∅）
        assert _norm_syllable("", "ê", "ê1") == ("", "ê", "1")
        assert _norm_syllable("", "ê", "ê3") == ("", "ê", "3")
        assert (ZERO_INITIAL, "ê", "1") in prof("欸") and (ZERO_INITIAL, "ê", "3") in prof("欸")
        # 分解形式 e+U+0302：pypinyin 认不出（韵母为空）→ None，由 ⑤ 兜住而不是静默留空串
        assert _norm_syllable("", "", "e\u0302") is None
        # 全库回归：修正后每个无调音节都有唯一的 (声母, 韵母)，即不再有碰撞/空韵母
        import pypinyin as _pypinyin_mod
        from pypinyin.pinyin_dict import pinyin_dict as _pd
        from pypinyin.style import convert as _cv

        _groups: dict[str, set[tuple[str, str]]] = {}
        _unrepresentable: list[str] = []
        for _v in _pd.values():
            for _one in _v.split(","):
                _d = _cv(_one, _pypinyin_mod.Style.TONE3, True)
                _base = _d[:-1] if _d[-1:].isdigit() else _d
                _tri = _norm_syllable(
                    _cv(_one, _pypinyin_mod.Style.INITIALS, True) or ZERO_INITIAL,
                    _cv(_one, _pypinyin_mod.Style.FINALS, True),
                    _d,
                )
                if _tri is None:
                    _unrepresentable.append(_base)
                else:
                    _groups.setdefault(_base, set()).add(_tri[:2])
        assert not _unrepresentable, sorted(set(_unrepresentable))  # 大陆字典不应有失败音节
        _coll = {k: sorted(v) for k, v in _groups.items() if len(v) > 1}
        assert not _coll, _coll
        assert len(_groups) == 426  # 无调音节总数（含 hm/hng/m/n/ng/yo）
        # 多音字：逐读音三元组（用 pinyin_dict + convert，不是把去重列表按下标配对）
        assert prof("行")[0] == ("x", "ing", "2")  # 主读音排第一
        assert ("h", "ang", "2") in prof("行") and ("x", "ing", "4") in prof("行")
        assert len(prof("行")) == 5
        assert prof("重")[0] == ("zh", "ong", "4") and ("ch", "ong", "2") in prof("重")
        assert is_polyphone("行") and is_polyphone("重") and not is_polyphone("鸭")
        triples, review = suggest_gold_phon("重庆")
        assert triples[0] == ("zh", "ong", "4") and review == [(1, "重")]
        assert phon_profile("5") is None and phon_profile("A") is None  # 非汉字

        # 本题最核心的一幕：猜「金腔鱼」→ 字属性 ⬜ 但三项音系属性全 🟩（同音异字）
        m = phon_marks("金腔鱼", "金枪鱼", cfg_pc)
        assert m[ATTR_CHAR] == [GREEN, GRAY, GREEN]
        assert m[ATTR_INITIAL] == m[ATTR_FINAL] == m[ATTR_TONE] == [GREEN, GREEN, GREEN]
        # 猜得比答案长 → 超出位是 ⬜，不是 NA
        assert phon_marks("金枪鱼呀", "金枪鱼", cfg_pc)[ATTR_CHAR][3] == GRAY
        # 答案位是非汉字 → 音系属性 NA（不能算错）
        m3 = phon_marks("5", "5", cfg_pc)
        assert m3[ATTR_CHAR] == [GREEN] and m3[ATTR_TONE] == [NA_MARK]
        # 冻结读音（多音字消歧）
        cfg_frozen = WordleConfig(
            feedback="phon_char", gold_phon=[("ch", "ong", "2"), ("q", "ing", "4")]
        )
        assert phon_marks("重庆", "重庆", cfg_frozen)[ATTR_FINAL] == [GREEN, GREEN]
        # 近音合并：z/zh
        assert phon_marks("子", "纸", WordleConfig(feedback="phon_near"))[ATTR_INITIAL] == [GREEN]
        assert phon_marks("子", "纸", cfg_pc)[ATTR_INITIAL] == [GRAY]
        # 舌尖韵母修正后：近音档下「资/知」不再被误判同音，「早/找」仍算同音
        _near = WordleConfig(feedback="phon_near")
        _m_zi = phon_marks("资", "知", _near)
        assert _m_zi[ATTR_INITIAL] == [GREEN] and _m_zi[ATTR_FINAL] == [GRAY]
        _m_zao = phon_marks("早", "找", _near)
        assert _m_zao[ATTR_INITIAL] == [GREEN] and _m_zao[ATTR_FINAL] == [GREEN]
        print("phon self-test: OK\n")

    # --- T1 后续轮自检（system 必须与第一轮逐字相同；反馈可读；绝不泄漏 gold） ---
    _it = {
        "id": "T1R",
        "hint_img": "H.jpg",
        "guess_img": "G.jpg",
        "hint_text": "这是金子",
        "guess_text": "金子拿枪对着一条鱼",
        "answer": "金枪鱼",
    }
    _stub = lambda p: "data:image/jpeg;base64,<STUB>"  # noqa: E731
    _t1_kw: dict[str, Any] = dict(
        input_mode="A", output_mode="json", image_encoder=_stub, hint_length=True
    )
    _sys1 = build_t1(_it, **_t1_kw)[0]["content"]
    assert "共 3 个字" in _sys1  # 字数提示在 system 里，后续轮会跟着重发
    _fbs = T1_RETRY_FEEDBACKS if _pypinyin() is not None else ("none", "bin", "tri")
    for _fb in _fbs:
        _r = build_t1_retry(_it, ["金腔鱼"], feedback=_fb, **_t1_kw)
        assert _r[0]["content"] == _sys1, f"{_fb}: 后续轮 system 与第一轮不一致"
        _ut = _first_text(_r[1]["content"])
        assert "金腔鱼" in _ut, f"{_fb}: 没说明上一轮答了什么"
        assert "共 3 个字" in _ut or "仍 3 个字" in _ut or "是 3 个字" in _ut, f"{_fb}: 缺字数提示"
        assert _it["answer"] not in _ut, f"{_fb}: 泄漏 gold"
        assert ("错误" in _ut and "再猜" in _ut) if _fb == "none" else "反馈说明" in _ut
    if _pypinyin() is not None:
        _u3 = _first_text(
            build_t1_retry(_it, ["金银铜", "金腔鱼"], feedback="phon_char", **_t1_kw)[1]["content"]
        )
        assert "第1轮" in _u3 and "第2轮" in _u3 and "第 3 轮" in _u3  # 逐轮列出 + 下一轮编号
    _bad_cases: list[Callable[[], Any]] = [
        lambda: build_t1_retry(_it, ["x"], feedback="nope", **_t1_kw),
        lambda: build_t1_retry(_it, [], **_t1_kw),
    ]
    for _bad in _bad_cases:
        try:
            _bad()
            raise AssertionError("应当抛 ValueError")
        except ValueError:
            pass
    print("T1 retry self-test: OK\n")

    item_t1 = {
        "id": "28",
        "hint_img": r"D:\Github\pun\data\release\images\28__hint.jpg",
        "guess_img": r"D:\Github\pun\data\release\images\28__guess.jpg",
        "hint_text": "这是金子",
        "guess_text": "金子拿枪对着一条鱼",
        "answer": "金枪鱼",
    }
    stub = lambda p: "data:image/jpeg;base64,<STUB>"  # noqa: E731  (演示用，不读盘)

    msgs = build_t1(item_t1, input_mode="A", output_mode="json", image_encoder=stub)
    print("=== T1 / A / json ===")
    print(json.dumps(msgs, ensure_ascii=False, indent=2)[:1200], "...\n")

    msgs_h = build_t1(
        item_t1,
        input_mode="A",
        output_mode="json",
        hint_length=True,
        hint_type="animal",
        image_encoder=stub,
    )
    print("=== T1 / A / json + 答案形状提示（字数+大类）===")
    print(msgs_h[0]["content"][-300:], "\n")

    if _pypinyin() is not None:
        print("=== T1 后续轮 / A / json / phon_char / 第2轮 ===")
        _m2 = build_t1_retry(
            item_t1, ["金腔鱼"], feedback="phon_char", image_encoder=stub, hint_length=True
        )
        print(_first_text(_m2[1]["content"]), "\n")

    item_t4 = {
        "id": "28",
        "hint_img": item_t1["hint_img"],
        "guess_img": item_t1["guess_img"],
        "positions": [
            {
                "gold": "金",
                "syllable": "jin",
                "candidates": ["金", "锦", "劲", "今", "斤", "进", "津", "晋"],
            },
            {"gold": "枪", "syllable": "qiang", "candidates": ["枪", "墙", "腔", "抢", "强", "羌"]},
            {
                "gold": "鱼",
                "syllable": "yu",
                "candidates": ["鱼", "予", "域", "昱", "榆", "豫", "预", "淤"],
            },
        ],
    }
    msgs = build_t4(item_t4, output_mode="letters", image_encoder=stub)
    print("=== T4 / letters (user content) ===")
    print(_first_text(msgs[1]["content"]), "\n")

    cfg = WordleConfig(max_turns=6, reveal_length=True, feedback="tri")
    msgs = build_wordle_turn(
        item_t1, history=["金银铜", "枪金鱼"], cfg=cfg, turn=3, image_encoder=stub
    )
    print("=== W / tri / turn 3 (user content) ===")
    print(_first_text(msgs[1]["content"]))
    print("=== W / 音系四项属性 / turn 3 (user content) ===")
    msgs_p = build_wordle_turn(
        item_t1,
        history=["金银铜", "金腔鱼"],
        cfg=WordleConfig(feedback="phon_char"),
        turn=3,
        image_encoder=stub,
    )
    print(_first_text(msgs_p[1]["content"]))
    print("\n--- system 规则（节选） ---")
    print(msgs_p[0]["content"][-460:], "\n")

    print("\n=== parse ===")
    print(parse_t1('```json\n{"hint_name":"金子","answer":"金枪鱼"}\n```', "json"))
    print(parse_wordle_guess("第3轮：枪金鱼", "guess"))
    print(parse_t4("金枪鱼", "letters"))


if __name__ == "__main__":
    _demo()
