"""D4 · 批元数据 shape 守卫 + NaN/Inf 守卫（P6 前置遗留）。

为什么在 P6 做: P3 已踩过「块表 padding 读脏块 / 末块半满未截断 → NaN」；P6 引入
块共享（ref_cnt>1）后，越界写会污染**其它请求**的 KV —— 症状是「别人的输出变了」，
不报错、不 NaN。守卫把这类静默算错变成立刻 raise。

本文件只测纯函数（不加载模型）。
运行: python -m pytest tests/test_p6_guards.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from nano_vllm.engine.core import check_finite, validate_batch_metadata


class TestValidateBatchMetadata:
    def _ok(self, **over):
        kwargs = dict(
            num_input_tokens=8,
            num_slot_mappings=8,
            kv_lens=[8, 4],
            block_tables=[[0, 1], [2]],
            block_size=16,
            qo_indptr_last=8,
            max_position_embeddings=32768,
        )
        kwargs.update(over)
        return kwargs

    def test_valid_passes(self):
        validate_batch_metadata(**self._ok())

    def test_slot_mapping_length_mismatch(self):
        with pytest.raises(ValueError, match="slot_mapping"):
            validate_batch_metadata(**self._ok(num_slot_mappings=7))

    def test_qo_indptr_last_mismatch(self):
        with pytest.raises(ValueError, match="qo_indptr"):
            validate_batch_metadata(**self._ok(qo_indptr_last=6))

    def test_kv_lens_and_block_tables_count_mismatch(self):
        with pytest.raises(ValueError, match="条数"):
            validate_batch_metadata(**self._ok(kv_lens=[8, 4, 1], block_tables=[[0, 1], [2]]))

    def test_kv_len_exceeds_block_table_capacity(self):
        """kv_len 超出块表容量会读到脏块 —— 必须拦下。"""
        with pytest.raises(ValueError, match="超出块表容量"):
            validate_batch_metadata(**self._ok(kv_lens=[8, 40], block_tables=[[0, 1], [2]]))

    def test_exactly_full_capacity_is_ok(self):
        """kv_len 恰好等于容量是合法的（末块刚好写满）。"""
        validate_batch_metadata(**self._ok(kv_lens=[16], block_tables=[[0]]))

    def test_exceeds_max_position_embeddings(self):
        with pytest.raises(ValueError, match="max_position_embeddings"):
            validate_batch_metadata(
                **self._ok(kv_lens=[64], block_tables=[[0, 1, 2, 3]], max_position_embeddings=32)
            )

    def test_max_position_none_skips_check(self):
        validate_batch_metadata(
            **self._ok(kv_lens=[64], block_tables=[[0, 1, 2, 3]], max_position_embeddings=None)
        )


class TestCheckFinite:
    def test_clean_tensor_passes(self):
        check_finite(torch.zeros(4), "x")

    def test_nan_raises(self):
        t = torch.zeros(4)
        t[2] = float("nan")
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            check_finite(t, "logits")

    def test_inf_raises(self):
        t = torch.zeros(4)
        t[0] = float("inf")
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            check_finite(t, "logits")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
