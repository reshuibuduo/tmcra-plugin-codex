"""Pinned retrieval profiles shared by the API and its real index workers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROFILE_FILE = Path(__file__).resolve().parent / "deploy" / "local-model-profiles.json"


def profiles():
    return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))["profiles"]


def profile_by_id(identity):
    for profile in profiles():
        if profile["id"] == identity:
            return profile
    raise ValueError("unknown local model profile")


def signature(profile):
    # A model change, pooling change or window-policy change requires reindexing.
    contract = {"embedding": profile["embedding"], "source_windows": "token-char-covered-v1",
                "long_slow_embedding": "normalized-window-mean-v1"}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def apply_local_profile(args):
    if os.getenv("TMCRA_DEPLOYMENT_MODE") != "local":
        return args
    profile = profile_by_id(os.environ["TMCRA_LOCAL_PROFILE"])
    embed = profile["embedding"]
    rerank = profile["reranker"]
    args.embedding_profile_id = profile["id"]
    args.embedding_index_signature = signature(profile)
    args.embedding_model = os.environ["TMCRA_EMBEDDING_MODEL"]
    args.cross_model = os.environ["TMCRA_CROSS_MODEL"]
    args.text_dim = embed["dimensions"]
    args.embedding_max_length = embed["model_max_tokens"]
    args.embedding_pooling = embed["pooling"]
    args.embedding_query_prefix = embed["query_prefix"]
    args.embedding_document_prefix = embed["document_prefix"]
    args.embedding_padding_side = embed["padding_side"]
    args.embedding_strict_max_length = True
    args.embedding_long_document_policy = "window_mean"
    args.reranker_mode = "fusion" if rerank["tmcra_fusion_checkpoint_compatible"] else "semantic-only"
    args.reranker_adapter = rerank["adapter"]
    args.cross_max_length = min(rerank["model_max_tokens"], 1280)
    args.cross_batch_size = 2 if args.device == "cpu" else 8
    args.batch_size = 4 if args.device == "cpu" else 8
    return args


def verify_index_identity(payload, args):
    expected = getattr(args, "embedding_index_signature", "")
    if expected and (payload.get("embedding_index_signature") != expected
                     or payload.get("text_dim") != args.text_dim):
        raise RuntimeError("index embedding identity differs; rebuild into a new generation before recall")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(model, directory):
    directory = Path(directory)
    for asset in model["weights"]:
        path = directory / asset["file"]
        if (not path.is_file() or path.stat().st_size != asset["bytes"]
                or sha256_file(path) != asset["sha256"]):
            raise RuntimeError(f"model asset is missing or failed SHA-256 verification: {asset['file']}")


QWEN_SYSTEM = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
              "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def qwen_rerank_windows(tokenizer, query, document, *, max_length):
    prefix = QWEN_SYSTEM + ("<Instruct>: Given a query about previous conversations, retrieve the relevant "
                            f"source passages.\n<Query>: {query}\n<Document>: ")
    head = tokenizer.encode(prefix, add_special_tokens=False)
    tail = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    body = tokenizer.encode(document, add_special_tokens=False)
    capacity = max_length - len(head) - len(tail)
    if capacity < 32:
        raise ValueError("reranker query leaves insufficient document context; shorten the query")
    step = max(1, capacity - min(192, capacity // 4))
    windows = []
    for start in range(0, max(1, len(body)), step):
        windows.append(head + body[start:start + capacity] + tail)
        if start + capacity >= len(body):
            break
    return windows
