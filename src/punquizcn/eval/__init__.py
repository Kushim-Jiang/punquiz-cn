"""评测子包：提示词套件、判分、以及实验驱动。

公开接口刻意收窄到最常用的几个函数；深层实现细节（反馈渲染的中间量、
提示词模板常量）仍可从 ``punquizcn.eval.prompts`` 直接导入。
"""

from __future__ import annotations

from punquizcn.eval.prompts import (
    PROMPT_VERSION,
    build_t1,
    build_t1_retry,
    build_t4,
    build_wordle_turn,
    corpus_path,
    encode_image_data_url,
    is_correct,
    parse_answer_only,
    parse_wordle_guess,
    phon_marks,
    render_feedback,
    wordle_marks,
)

__all__ = [
    "PROMPT_VERSION",
    "build_t1",
    "build_t1_retry",
    "build_t4",
    "build_wordle_turn",
    "corpus_path",
    "encode_image_data_url",
    "is_correct",
    "parse_answer_only",
    "parse_wordle_guess",
    "phon_marks",
    "render_feedback",
    "wordle_marks",
]
