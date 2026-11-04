# SPDX-License-Identifier: MIT
"""Direct A8W4 attention through ATOM's quantized producers and CSR builder."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="the V4.1 cache and CSR kernels need Triton")
pytest.importorskip("aiter", reason="V4.1 device attention needs AITER")

from atom.model_ops.attentions.deepseek_v41.a8w4_attention import (
    select_a8w4_backend,
    sparse_attn_v41_a8w4_paged_2src,
)
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    MAIN_FP4,
    V41PoolGeometry,
)
from atom.model_ops.blockscale import quantize_fp4, quantize_fp8
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


def mxfp8(planes):
    data, scale = planes
    values = data.view(torch.float8_e4m3fn).float()
    return values * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(
        32, -1
    )


def nvfp4(planes):
    data, scale = planes
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)


def make_case(heads, decode, ratio=2):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    backend = select_a8w4_backend(heads, 512, 512**-0.5)
    if backend is None:
        pytest.skip("This AITER/device combination has no A8W4 backend")
    torch.manual_seed(7407 + heads + ratio)
    geo = V41PoolGeometry(
        1,
        ((0, 2),),
        32,
        8,
        512,
        128,
        kv_dtype="fp4",
        speculative_tokens=3,
        layer_ratios=(0, 2),
        index_topk=4,
    )
    cache = PagedAttentionCache(geo, 7, 4, "cuda", max_tokens=4)
    data, scale = cache.main_view(0)
    raw = torch.randn(data.shape[0], 512, device="cuda", dtype=torch.bfloat16)
    packed, scales = quantize_fp4(raw, **MAIN_FP4)
    data.copy_(packed.view(torch.uint8))
    scale.copy_(scales.view(torch.uint8))
    data, scale = cache.ring_view(0)
    raw = torch.randn(data.shape, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(raw)
    data.copy_(values.view(torch.uint8))
    scale.copy_(scales.view(torch.uint8))
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    step = BatchStep(
        (RequestSpan(0, 10, 0, 2, 2), RequestSpan(1, 31, 2, 2, 0)),
        tensor([10, 11, 31, 32]),
        tensor([0, 2, 4]),
        tensor([2, 0]),
        tensor([0, 0, 1, 1]),
        tensor([[2, 1, 0], [4, 3, 5]]),
        scheduled=4,
        max_q_len=2,
        is_prefill=not decode,
        tentative=decode,
        swa_replay_start=tensor([0, 0]),
        swa_cached_until=tensor([0, 0]),
    )
    spec = SimpleNamespace(
        layer_id=0,
        kv_owner=0 if ratio else None,
        ratio=ratio,
        topk_owner=0,
        index_group_start=0,
        index_group_size=1,
    )
    return SimpleNamespace(
        cache=cache,
        step=step,
        spec=spec,
        backend=backend,
        query=torch.randn(1, 4, heads, 512, device="cuda", dtype=torch.bfloat16),
        kv=torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16),
        selected=tensor([[0, 1, 2, 3]] * 4),
        rope=RotaryEmbedding(64, 128, base=10000).cuda(),
        sink=torch.linspace(-1, 1, heads, device="cuda"),
    )


def run_case(case, splits=None):
    cache, step = case.cache, case.step
    step.begin_forward()
    step.selected[0] = case.selected
    q8 = cache.query_fp8(case.query, a8w4=True)
    source, pending = cache.rope_quant_window(
        0, case.query, case.kv, case.rope, step, q8, a8w4=True
    )
    pi, pp, wi, wp = cache.attention_indices(case.spec, step, a8w4=True)
    if step.decode:
        cache.write_window(0, pending, step)
        source = cache.ring_view(0)
    query = tuple(value.flatten(0, 1) for value in q8)
    main = cache.main_view(case.spec.kv_owner)
    result = torch.empty_like(case.query.flatten(0, 1))
    sparse_attn_v41_a8w4_paged_2src(
        query,
        main,
        pi,
        pp,
        source,
        wi,
        wp,
        case.sink,
        512**-0.5,
        backend=case.backend,
        out=result,
        num_kv_splits=splits,
    )
    if not step.decode:
        cache.write_window(0, pending, step)
    return result, (query, main, source, (pi, pp, wi, wp))


def reference(case, inputs):
    query, main, source, (pi, pp, wi, wp) = inputs
    q, csa, swa = mxfp8(query), nvfp4(main), mxfp8(source)
    pp, wp = pp.tolist(), wp.tolist()
    out = torch.zeros_like(q)
    for row in range(q.shape[0]):
        keys = torch.cat(
            (
                csa[pi[pp[row] : pp[row + 1]].long()],
                swa[wi[wp[row] : wp[row + 1]].long()],
            )
        )
        if keys.shape[0]:
            logits = q[row] @ keys.T * (512**-0.5)
            weights = torch.cat((logits, case.sink[:, None]), -1).softmax(-1)[:, :-1]
            out[row] = weights @ keys
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("heads", [2, 16, 32, 64])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("splits", [1, 4, None])
def test_a8w4_matches_dequantized_attention(heads, decode, ratio, splits):
    case = make_case(heads, decode, ratio)
    output, inputs = run_case(case, splits)
    expected = reference(case, inputs)
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("decode", [False, True])
def test_a8w4_graph_replay_changes_live_rows(decode):
    case = make_case(64, decode)
    run_case(case, 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, inputs = run_case(case, 1)
    for active in (2, 1, 2):
        case.query.normal_()
        case.kv.normal_()
        case.step.batch_ids.copy_(
            torch.tensor([0, 0, 1, 1] if active == 2 else [0, 0, -1, -1], device="cuda")
        )
        case.step.cu_seqlens_q[-1] = active * 2
        case.step.slots[-1] = 0 if active == 2 else -1
        graph.replay()
        torch.testing.assert_close(
            output, reference(case, inputs), rtol=0.03, atol=0.02
        )
