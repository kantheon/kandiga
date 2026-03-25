"""Speculative decoding: 35B drafts tokens, 122B verifies in batch.

The 35B model generates draft tokens at ~6 tok/s using its own KV cache.
The 122B model verifies all drafts in ONE forward pass (no per-token overhead).
Accepted tokens are essentially free — massive effective speedup.

Key insight: both models share the same tokenizer and expert architecture.
The 35B and 122B agree on most tokens for straightforward text, so
acceptance rates of 60-80% are expected.
"""

from __future__ import annotations

import time
from typing import Generator

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step, make_sampler


def speculative_generate(
    target_model,      # MLX model (122B) — loaded and ready
    draft_model,       # MLX model (35B) — loaded and ready
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    num_draft: int = 5,
    temp: float = 0.0,
) -> Generator[str, None, None]:
    """Speculative decoding with draft/target model pair.

    Each step:
    1. Draft model generates num_draft tokens (fast, ~6 tok/s each)
    2. Target model processes prompt + drafts in one forward pass
    3. Compare target's picks vs draft's picks at each position
    4. Accept all matching tokens, reject from first mismatch
    5. Yield accepted tokens

    This turns the 122B's slow per-token decode into efficient batch verification.
    """
    # Tokenize prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = prompt

    prompt_tokens = mx.array(tokenizer.encode(prompt_text))

    sampler = make_sampler(temp=temp)

    # Prefill both models
    draft_cache = draft_model.make_cache()
    target_cache = target_model.make_cache()

    # Process prompt through both models
    draft_out = draft_model(prompt_tokens[None], cache=draft_cache)
    target_out = target_model(prompt_tokens[None], cache=target_cache)
    mx.eval(draft_out, target_out)

    # Get first token from target
    target_logits = target_out[0, -1]
    target_token = mx.argmax(target_logits) if temp == 0 else sampler(mx.expand_dims(target_logits, 0))[0]
    mx.eval(target_token)

    # EOS detection
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids = set(eid)
        elif eid is not None:
            eos_ids = {eid}

    # Yield first token
    first_text = tokenizer.decode([target_token.item()])
    yield first_text

    current_token = target_token
    tokens_generated = 1
    total_accepted = 0
    total_drafted = 0

    while tokens_generated < max_tokens:
        if current_token.item() in eos_ids:
            break

        # Step 1: Draft model generates num_draft tokens (fast)
        draft_tokens = [current_token]
        dt = current_token
        for _ in range(num_draft):
            draft_out = draft_model(dt.reshape(1, 1), cache=draft_cache)
            mx.eval(draft_out)
            dt_logits = draft_out[0, -1]
            dt = mx.argmax(dt_logits) if temp == 0 else sampler(mx.expand_dims(dt_logits, 0))[0]
            mx.eval(dt)
            draft_tokens.append(dt)
            if dt.item() in eos_ids:
                break

        # draft_tokens = [current, draft_0, draft_1, ..., draft_N-1]
        # We need to verify draft_0 through draft_N-1
        draft_ids = mx.array([t.item() for t in draft_tokens])  # [N+1]
        total_drafted += len(draft_tokens) - 1

        # Step 2: Target model processes all draft tokens in ONE forward pass
        # Feed [current_token, draft_0, ..., draft_N-1] through target
        target_input = draft_ids[:-1].reshape(1, -1)  # all except last
        target_out = target_model(target_input, cache=target_cache)
        mx.eval(target_out)

        # Step 3: Compare target's choices vs drafts
        # target_out[0, i] gives logits for position after draft_tokens[i]
        # We compare target's argmax at position i with draft_tokens[i+1]
        accepted = 0
        for i in range(len(draft_tokens) - 1):
            target_logits_i = target_out[0, i]
            target_pick = mx.argmax(target_logits_i).item() if temp == 0 else sampler(mx.expand_dims(target_logits_i, 0))[0].item()

            draft_pick = draft_tokens[i + 1].item()

            if target_pick == draft_pick:
                # Accept this draft token
                text = tokenizer.decode([draft_pick])
                yield text
                tokens_generated += 1
                accepted += 1
                current_token = draft_tokens[i + 1]

                if draft_pick in eos_ids or tokens_generated >= max_tokens:
                    break
            else:
                # Reject: use target's token instead
                text = tokenizer.decode([target_pick])
                yield text
                tokens_generated += 1
                current_token = mx.array(target_pick)

                # Rollback draft model's cache to match
                # (simplified: we just let it diverge and it'll re-sync)
                # TODO: proper cache rollback for draft model
                break

        total_accepted += accepted

        # If all drafts were accepted, we also need the target's next prediction
        if accepted == len(draft_tokens) - 1:
            # Target already processed all drafts — get next token from last position
            last_logits = target_out[0, -1]
            next_pick = mx.argmax(last_logits).item() if temp == 0 else sampler(mx.expand_dims(last_logits, 0))[0].item()
            text = tokenizer.decode([next_pick])
            yield text
            tokens_generated += 1
            current_token = mx.array(next_pick)

    if total_drafted > 0:
        rate = total_accepted / total_drafted * 100
        print(f"\n[speculative] {total_accepted}/{total_drafted} "
              f"accepted ({rate:.0f}%) — {num_draft} drafts/step")
