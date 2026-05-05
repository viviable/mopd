from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd


def _to_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_plain(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _prompt_to_messages(prompt: Any) -> list[dict[str, Any]]:
    prompt = _to_plain(prompt)
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    return [{"role": "user", "content": str(prompt)}]


def convert(src: str, dst: str) -> None:
    df = pd.read_parquet(src)
    rows = []

    for _, row in df.iterrows():
        base_messages = _prompt_to_messages(row["prompt"])
        responses = _to_plain(row["responses"])
        if not isinstance(responses, list):
            responses = [responses]

        for response in responses:
            if response is None:
                continue
            response = str(response).strip()
            if not response:
                continue
            rows.append({"messages": base_messages + [{"role": "assistant", "content": response}]})

    if not rows:
        raise RuntimeError(f"No non-empty teacher responses found in {src}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_parquet(dst)
    print(f"Wrote {len(out)} SFT examples to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GKD teacher rollout parquet to multiturn SFT parquet.")
    parser.add_argument("--train-src", required=True)
    parser.add_argument("--val-src", required=True)
    parser.add_argument("--train-dst", required=True)
    parser.add_argument("--val-dst", required=True)
    args = parser.parse_args()

    convert(args.train_src, args.train_dst)
    convert(args.val_src, args.val_dst)


if __name__ == "__main__":
    main()
