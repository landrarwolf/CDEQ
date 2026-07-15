from types import SimpleNamespace

import torch

from llm_cdeq.verify_jacobi import greedy_ar_block, vanilla_jacobi_endpoint


class IncrementCausalModel:
    def __call__(self, *, input_ids, use_cache):
        assert use_cache is False
        vocabulary = 16
        targets = (input_ids + 1) % vocabulary
        logits = torch.full((*input_ids.shape, vocabulary), -1.0)
        logits.scatter_(-1, targets.unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits)


def test_vanilla_jacobi_matches_no_cache_greedy_ar():
    model = IncrementCausalModel()
    prefix = torch.tensor([[1, 2]])
    _, ar_endpoint = greedy_ar_block(model, prefix, block_size=4)
    jacobi_endpoint, iterations = vanilla_jacobi_endpoint(
        model, prefix, torch.tensor([[3, 0, 0, 0]])
    )
    assert ar_endpoint.tolist() == [[3, 4, 5, 6]]
    assert torch.equal(jacobi_endpoint, ar_endpoint)
    assert 1 <= iterations <= 5
