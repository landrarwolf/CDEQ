import os
from pathlib import Path

import pytest
import torch

from llm_cdeq.cllm_step import OfficialCLLMSingleStep, parameter_checksum
from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq.prepare_cllm_states import normalize_initial_example
from llm_cdeq.runtime import iter_json_array
from llm_cdeq.wrapped_model import WrappedCLLM


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OFFICIAL_CLLM_TESTS") != "1",
    reason="requires the staged official 7B checkpoint and an otherwise empty GPU",
)


def test_official_7b_wrapped_equivalence_cache_and_freeze():
    model_path = Path(
        os.environ.get(
            "OFFICIAL_CLLM_PATH",
            "/home/ljc/models/cllm/consistency-llm-7b-math",
        )
    )
    data_path = Path(
        os.environ.get(
            "CLLM_INITIAL_STATE_DATA",
            "/home/ljc/data/cllm/cleaned_gsm8k_train_jacobian16_augTrue_labels_True_max_seq_len_256.json",
        )
    )
    operator = OfficialCLLMSingleStep.from_pretrained(
        model_path,
        block_size=16,
        attention_backend="sdpa",
        device="cuda",
    )
    example = normalize_initial_example(next(iter_json_array(data_path)), 16)
    prompt = torch.tensor([example["prompt_ids"]], dtype=torch.long, device="cuda")
    current = torch.tensor(
        [example["initial_tokens"]], dtype=torch.long, device="cuda"
    )
    prefill = operator.prefill(prompt)
    assert torch.equal(current[:, :1], prefill.first_token)

    corrector = TransformerResidualCorrector().to(device="cuda", dtype=torch.bfloat16)
    wrapped = WrappedCLLM(operator, corrector)
    zero = wrapped(prefill, current, torch.tensor(0.0, device="cuda"), round_index=0)
    assert torch.equal(zero.hidden, zero.base.canonical_hidden)
    assert torch.equal(zero.logits, zero.base.logits)
    assert torch.equal(zero.tokens, zero.base.tokens)
    assert zero.base.prompt_cache_lengths_before == zero.base.prompt_cache_lengths_after

    with torch.no_grad():
        torch.nn.init.normal_(corrector.up.weight, std=1e-3)
    terminal = wrapped(prefill, current, torch.tensor(5.0, device="cuda"), round_index=0)
    disabled = wrapped(
        prefill,
        current,
        torch.tensor(0.0, device="cuda"),
        round_index=0,
        disable_adapter=True,
    )
    for output in (terminal, disabled):
        assert torch.equal(output.hidden, output.base.canonical_hidden)
        assert torch.equal(output.logits, output.base.logits)
        assert torch.equal(output.tokens, output.base.tokens)
    assert not any(parameter.requires_grad for parameter in operator.model.parameters())

    # The expensive full checksum is opt-in but is used by the acceptance run.
    if os.environ.get("RUN_FULL_BACKBONE_CHECKSUM") == "1":
        checksum_before = parameter_checksum(operator.model)
        optimizer = torch.optim.AdamW(corrector.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        random_state = torch.randn(
            1, 16, 4096, device="cuda", dtype=torch.bfloat16
        )
        corrector(random_state, torch.tensor(0.0, device="cuda")).float().square().mean().backward()
        optimizer.step()
        checksum_after = parameter_checksum(operator.model)
        assert checksum_before == checksum_after
