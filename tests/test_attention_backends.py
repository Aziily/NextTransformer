import torch

from launch.attention_backends import sliding_window, window_mask


def test_sliding_window_matches_the_sssl_short_layers():
    assert sliding_window(2048) == 512
    assert sliding_window(1000) == 256


def test_window_mask_is_causal_and_bounded():
    mask = window_mask(torch, 8, 2, torch.device("cpu"))
    assert mask.tolist()[5] == [False, False, False, True, True, True, False, False]
    assert not mask.triu(1).any()


def test_full_window_needs_no_mask():
    assert window_mask(torch, 8, None, torch.device("cpu")) is None
    assert window_mask(torch, 8, 7, torch.device("cpu")) is None
