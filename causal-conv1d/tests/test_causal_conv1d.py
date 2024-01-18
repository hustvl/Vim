# Copyright (C) 2023, Tri Dao.

import math

import torch
import pytest

from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref


@pytest.mark.parametrize("channel_last", [False, True])
# @pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [2])
@pytest.mark.parametrize(
    "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
def test_causal_conv1d(seqlen, width, has_bias, silu_activation, itype, channel_last):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    # batch_size = 1
    dim = 4096 + 32  # Try dim not divisible by 64
    # dim = 64
    if not channel_last:
        x = torch.randn(batch_size, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch_size, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, activation=activation)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, activation=activation)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [False])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [2])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    # batch_size = 1
    # dim = 64
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    conv_state = torch.randn(batch_size, dim, width, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


# @pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize('channel_last', [True])
# @pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.bfloat16])
# @pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize('silu_activation', [True])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize('has_bias', [True])
# @pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize(
    # "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
    "seqlen", [2048]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
def test_causal_conv1d_race_condition(seqlen, width, has_bias, silu_activation, itype, channel_last):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    # batch_size = 1
    dim = 4096 + 32  # Try dim not divisible by 64
    # dim = 64
    if not channel_last:
        x = torch.randn(batch_size, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch_size, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    activation = None if not silu_activation else "silu"
    out0 = causal_conv1d_fn(x, weight, bias, activation=activation)
    g = torch.randn_like(out0)
    dx0, dw0, db0 = torch.autograd.grad(out0, (x, weight, bias), g)
    dw_atol = 1e-4
    db_atol = 1e-4

    for i in range(10000):
        out = causal_conv1d_fn(x, weight, bias, activation=activation)
        dx, dw, db = torch.autograd.grad(out, (x, weight, bias), g)
        dw_equal = torch.allclose(dw, dw0, atol=dw_atol)
        # if not dw_equal:
        #     breakpoint()
        if has_bias:
            db_equal = torch.allclose(db, db0, atol=db_atol)
            # if not db_equal:
            #     breakpoint()
        assert torch.equal(out, out0)
        assert torch.equal(dx, dx0)
        assert dw_equal
        if has_bias:
            assert dw_equal
