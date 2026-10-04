"""端到端冒烟测试：提示词构建、判分三级、防泄漏护栏、分层抽样。

用 ``data/sample`` 里的 10 条开放样本跑通全部主要代码路径。
**不需要推理服务器**——只构造 messages，不发请求。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from punquizcn.eval import prompts
from punquizcn.eval.runner import Item, stratified_sample

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
ITEMS_PATH = SAMPLE_DIR / "sample_items.jsonl"
IMAGE_DIR = SAMPLE_DIR / "images"

ALL_MODES = list(prompts.T1_INPUT_MODES)
ALL_OUTPUTS = list(prompts.T1_OUTPUT_MODES)


def _stub_encoder(_path: str | Path) -> str:
    """替身编码器：不真的读图，返回固定的 data URL。

    这样测试既快又不依赖 Pillow，仍然能验证 messages 的**结构**
    （图片数量、文本位置）是否正确。
    """
    return "data:image/jpeg;base64,AAAA"


@pytest.fixture(scope="module")
def raw_items() -> list[dict[str, Any]]:
    """读出 10 条样本，并补上 ``hint_img``/``guess_img`` 两个图片路径字段。"""
    if not ITEMS_PATH.is_file():
        pytest.skip(f"样本文件不在：{ITEMS_PATH}")
    items = [
        json.loads(line)
        for line in ITEMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for it in items:
        it["hint_img"] = str(IMAGE_DIR / f"{it['id']}__hint.jpg")
        it["guess_img"] = str(IMAGE_DIR / f"{it['id']}__guess.jpg")
    return items


@pytest.fixture(scope="module")
def sample_items(raw_items: list[dict[str, Any]]) -> list[Item]:
    """把原始 dict 样本转成 runner 的 Item 对象。"""
    return [Item.from_json(it) for it in raw_items]


def _count_images(messages: list[dict[str, Any]]) -> int:
    """数第一条多模态消息里的图片块数量。"""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            return sum(1 for blk in content if blk.get("type") == "image_url")
    return 0


# ---------------------------------------------------------------------------
# 1. 样本本身
# ---------------------------------------------------------------------------
def test_sample_images_present(raw_items: list[dict[str, Any]]) -> None:
    """样本的每一条都必须配齐 hint/guess 两张图，否则后面的用例会静默变弱。"""
    missing = [
        it["id"]
        for it in raw_items
        if not Path(it["hint_img"]).is_file() or not Path(it["guess_img"]).is_file()
    ]
    assert not missing, f"样本图片缺失：{missing}"
    assert len(raw_items) == 13, f"样本条数应为 13，实际 {len(raw_items)}"


def test_sample_ids_are_unique(sample_items: list[Item]) -> None:
    """id 重复会让续跑逻辑误判「已完成」，必须唯一。"""
    ids = [it.id for it in sample_items]
    assert len(ids) == len(set(ids)), f"id 重复：{ids}"


def test_sample_has_multiple_answer_lengths(sample_items: list[Item]) -> None:
    """样本应覆盖多种答案长度，才能真的测到分层的意义。"""
    lengths = {len(it.answer) for it in sample_items}
    assert len(lengths) >= 3, f"答案字数分布过窄：{lengths}"


# ---------------------------------------------------------------------------
# 2. 四个输入档的图片数量契约
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mode", "expected_images"),
    [("A", 2), ("B", 2), ("C", 2), ("D", 0)],
)
def test_input_mode_image_count(
    raw_items: list[dict[str, Any]], mode: str, expected_images: int
) -> None:
    """A/B/C 都给两张原图；D 是纯文字档，必须一张图都不带。"""
    assert mode in prompts.T1_INPUT_MODES
    for it in raw_items[:3]:
        messages = prompts.build_t1(
            it,
            input_mode=mode,
            output_mode="answer_only",
            image_encoder=_stub_encoder,
            hint_length=None,
        )
        assert _count_images(messages) == expected_images, f"{it['id']} / 输入档 {mode}"


def test_output_mode_json_only_changes_instruction(
    raw_items: list[dict[str, Any]],
) -> None:
    """两种输出档的图片数应一致，差异只在文本指令里。"""
    it = raw_items[0]
    answer_only = prompts.build_t1(
        it, input_mode="A", output_mode="answer_only", image_encoder=_stub_encoder
    )
    json_mode = prompts.build_t1(
        it, input_mode="A", output_mode="json", image_encoder=_stub_encoder
    )
    assert _count_images(answer_only) == _count_images(json_mode) == 2
    assert prompts.flatten_messages(answer_only) != prompts.flatten_messages(json_mode)


# ---------------------------------------------------------------------------
# 3. 判分三级
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("guess", "gold", "expected"),
    [
        # exact：逐字相同
        ("葫芦娃", "葫芦娃", "exact"),
        # homophone：逐位声母+韵母+声调全同，只是汉字不同
        ("圆周律", "圆周率", "homophone"),
        # no：读音也不同
        ("完全无关", "葫芦娃", "no"),
        ("圆周五", "圆周率", "no"),
        # no：长度不同（判定要求逐位对齐）
        ("葫芦", "葫芦娃", "no"),
        # empty：模型什么都没给
        ("", "葫芦娃", "empty"),
        ("   ", "葫芦娃", "empty"),
    ],
)
def test_is_correct_levels(guess: str, gold: str, expected: str) -> None:
    ok, why = prompts.is_correct(guess, gold)
    assert why == expected, f"猜「{guess}」对「{gold}」判定为 {why}，期望 {expected}"
    assert ok == (expected in ("exact", "homophone"))


def test_is_correct_ignores_whitespace() -> None:
    """归一化应吃掉空白与全角空格。"""
    assert prompts.is_correct(" 葫芦 娃 ", "葫芦娃")[0] is True


def test_is_correct_near_tier() -> None:
    """near = 逐位声母+韵母相同、但至少一位声调不同。

    例：「马」ma3 与「妈」ma1 声韵同、调不同。
    """
    ok, why = prompts.is_correct("妈妈", "马妈")
    assert (ok, why) == (True, "near")


def test_is_correct_tone_difference_is_not_homophone() -> None:
    """只要有调不同就不能算 homophone（homophone 是主口径，必须严格）。"""
    _, why = prompts.is_correct("马", "妈")
    assert why == "near"


# ---------------------------------------------------------------------------
# 4. 防泄漏护栏：不误报
# ---------------------------------------------------------------------------
def test_leak_guard_passes_on_all_combinations(
    raw_items: list[dict[str, Any]],
) -> None:
    """10 条 × 4 输入档 × 2 输出档，护栏全过——证明它不误报。"""
    checked = 0
    for it in raw_items:
        for mode in ALL_MODES:
            for out in ALL_OUTPUTS:
                messages = prompts.build_t1(
                    it,
                    input_mode=mode,
                    output_mode=out,
                    image_encoder=_stub_encoder,
                    hint_length=True,
                )
                prompts.assert_no_leak(messages, it, mode, 1, prev_answers=[])
                checked += 1
    assert checked == len(raw_items) * len(ALL_MODES) * len(ALL_OUTPUTS)


def test_leak_guard_allows_model_guessing_right(
    raw_items: list[dict[str, Any]],
) -> None:
    """模型自己上一轮猜中答案，不算提示词泄漏（prev_answers 会被剥离）。"""
    it = raw_items[0]
    gold = it["answer"]
    messages = [
        {"role": "system", "content": "你是解题者。"},
        {"role": "user", "content": f"第1轮你的回答：「{gold}」——错误，再猜。"},
    ]
    # 不该抛异常
    prompts.assert_no_leak(messages, it, "A", 2, prev_answers=[gold])


# ---------------------------------------------------------------------------
# 5. 防泄漏护栏：必须抓住真泄漏
# ---------------------------------------------------------------------------
def test_leak_guard_catches_plain_answer(raw_items: list[dict[str, Any]]) -> None:
    it = raw_items[0]
    messages = [{"role": "system", "content": "答案是 " + it["answer"]}]
    with pytest.raises(AssertionError):
        prompts.assert_no_leak(messages, it, "X", 1, prev_answers=[])


def test_leak_guard_catches_template_phrase() -> None:
    """回归：模板里「不要用三反引号代码块包裹」曾误伤答案是「代码」的题（id 10834）。

    现在 strip_prompt_boilerplate 会把模板常量先剥掉，所以这条必须通过。
    """
    item = {
        "id": 10834,
        "answer": "代码",
        "hint_text": "",
        "guess_text": "",
        "hint_img": "",
        "guess_img": "",
    }
    messages = prompts.build_t1(
        item, input_mode="B", output_mode="json", image_encoder=_stub_encoder
    )
    prompts.assert_no_leak(messages, item, "B", 1, prev_answers=[])


# ---------------------------------------------------------------------------
# 6. 分层抽样
# ---------------------------------------------------------------------------
def test_stratified_sample_is_reproducible(sample_items: list[Item]) -> None:
    first = stratified_sample(sample_items, 2, seed=20261004)
    second = stratified_sample(sample_items, 2, seed=20261004)
    assert [x.id for x in first] == [x.id for x in second]


def test_stratified_sample_seed_changes_pick(sample_items: list[Item]) -> None:
    """不同 seed 抽到不同批题——档间比较才需要固定 seed。"""
    a = stratified_sample(sample_items, 2, seed=20261004)
    b = stratified_sample(sample_items, 2, seed=999)
    assert [x.id for x in a] != [x.id for x in b]


def test_stratified_sample_respects_per_class(sample_items: list[Item]) -> None:
    picked = stratified_sample(sample_items, 1, seed=1)
    counts: dict[str, int] = {}
    for it in picked:
        key = str(len(it.answer))
        counts[key] = counts.get(key, 0) + 1
    assert all(n == 1 for n in counts.values()), counts


# ---------------------------------------------------------------------------
# 7. 解析器
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 纯答案
        ("葫芦娃", "葫芦娃"),
        # 取最后一行
        ("让我想想…\n葫芦娃", "葫芦娃"),
        # 去掉行首序号
        ("1. 葫芦娃", "葫芦娃"),
        ("3、葫芦娃", "葫芦娃"),
        # 去掉包裹的反引号
        ("`葫芦娃`", "葫芦娃"),
        # 多行时只看最后一行
        ("第一行解释\n第二行解释\n葫芦娃", "葫芦娃"),
    ],
)
def test_parse_answer_only(raw: str, expected: str) -> None:
    assert prompts.parse_answer_only(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_parse_answer_only_returns_empty_on_blank(raw: str) -> None:
    assert prompts.parse_answer_only(raw) == ""


def test_parse_answer_only_gives_up_on_overlong_line() -> None:
    """超长行不该被当成答案（那多半是模型没听话，在写解释）。"""
    long_text = "这是一段很长的解释没有任何答案" * 3
    assert len(prompts.parse_answer_only(long_text)) > 12


# ---------------------------------------------------------------------------
# 8. Wordle 反馈
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("guess", "gold", "expected"),
    [
        ("金枪鱼", "金枪鱼", [prompts.GREEN] * 3),
        ("枪金鱼", "金枪鱼", [prompts.YELLOW, prompts.YELLOW, prompts.GREEN]),
        ("金银铜", "金枪鱼", [prompts.GREEN, prompts.GRAY, prompts.GRAY]),
        # 重复字只标一次
        ("金金金", "金枪鱼", [prompts.GREEN, prompts.GRAY, prompts.GRAY]),
        # 长度不足 / 超出
        ("金银", "金枪鱼", [prompts.GREEN, prompts.GRAY]),
        ("金银铜铁", "金枪鱼", [prompts.GREEN, prompts.GRAY, prompts.GRAY, prompts.GRAY]),
    ],
)
def test_wordle_marks(guess: str, gold: str, expected: list[str]) -> None:
    assert prompts.wordle_marks(guess, gold) == expected


def test_wordle_marks_distinguishes_bin_from_tri() -> None:
    """回归：选一个「字都在答案里但位置全错」的猜测。

    这样才能区分 binary（全 ⬜）与 ternary（全 🟨）反馈——
    若用猜中或完全无关的猜测，两档结果相同，测了等于没测。
    """
    guess, gold = "枪鱼金", "金枪鱼"
    assert prompts.wordle_marks(guess, gold) == [
        prompts.YELLOW,
        prompts.YELLOW,
        prompts.YELLOW,
    ]
    cfg_bin = prompts.WordleConfig(feedback="bin").validate()
    cfg_tri = prompts.WordleConfig(feedback="tri").validate()
    rendered_bin = prompts.render_feedback(guess, gold, cfg_bin)
    rendered_tri = prompts.render_feedback(guess, gold, cfg_tri)
    assert rendered_bin != rendered_tri
