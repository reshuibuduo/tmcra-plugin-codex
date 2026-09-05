from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import networkx as nx

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for Tri-Maze neural training: {exc}")

from .concept_graph import ConceptGraph
from .concept_memory import ConceptMemory
from .maze_engine import TriMazeEngine
from .policy_dataset import (
    CurriculumConfig,
    EpisodicBatchSampler,
    PolicyStepDataset,
    PolicyStepRecord,
    PolicyVocabulary,
    build_domain_sampling_weights,
    build_path_stub,
    build_runtime_step_record,
    filter_curriculum_records,
    load_policy_records,
    policy_collate_fn,
    serialize_policy_records,
)
from .policy_network import (
    EdgePolicy,
    PolicyModelConfig,
    candidate_contrastive_loss,
    hard_negative_margin_loss,
    masked_cross_entropy,
)
from .tri_maze_supervision import load_supervision_rows, normalize_supervision_row, summarize_supervision_rows


DEFAULT_MEMORY_FILE = "data/concept_memory.json"
DEFAULT_SUPERVISION_GLOBS = ("data/tri_maze_datasets/exports/*tri_maze_supervision*.jsonl",)


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 1.0
    branch_factor: int = 2
    revisit_probability: float = 0.2
    train_ratio: float = 0.9
    patience: int = 8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    amp: bool = True
    model_type: str = "v2"
    embedding_dim: int = 64
    relation_embedding_dim: int = 16
    domain_embedding_dim: int = 8
    trunk_dims: tuple[int, ...] = (128, 256, 128)
    dropout: float = 0.1
    history_size: int = 4
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    domain_balance: bool = True
    episodic: bool = False
    multitask: bool = True
    contrastive_weight: float = 0.08
    hard_negative_weight: float = 0.12
    aux_weight: float = 0.15
    domain_adapt: bool = False
    domain_adapt_weight: float = 0.03
    zero_shot_split: str = "none"
    cache_dir: str = ""
    resume: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trunk_dims"] = list(self.trunk_dims)
        payload["curriculum"] = asdict(self.curriculum)
        return payload


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _stable_score(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _clamp_score(value: Any, default: float = 0.8) -> float:
    try:
        return max(0.1, min(1.0, float(value)))
    except Exception:
        return default


def _weight_from_path(path_record: Dict[str, Any]) -> float:
    base = _clamp_score(path_record.get("score", 0.8), default=0.8)
    try:
        uses = max(1, int(path_record.get("uses", 1) or 1))
    except Exception:
        uses = 1
    return base * math.log1p(uses)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_supervision_files(
    *,
    repo_root: Path | None = None,
    files: Sequence[str] | None = None,
    globs: Sequence[str] | None = None,
) -> list[Path]:
    root = repo_root or _repo_root()
    discovered: list[Path] = []
    for item in files or []:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            discovered.append(candidate)
    for pattern in globs or DEFAULT_SUPERVISION_GLOBS:
        for child in sorted(root.glob(pattern)):
            if child.is_file():
                discovered.append(child)
    ordered: list[Path] = []
    seen = set()
    for path in discovered:
        marker = str(path.resolve()).casefold()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(path)
    return ordered


def load_all_supervision_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_supervision_rows(path))
    return rows


def build_training_graph(
    memory: ConceptMemory,
    *,
    supervision_rows: Sequence[dict[str, Any]] | None = None,
    graph_json_path: str | None = None,
) -> nx.DiGraph:
    if graph_json_path:
        payload = json.loads(Path(graph_json_path).read_text(encoding="utf-8"))
        concept_graph = ConceptGraph()
        concept_graph.import_json(payload)
        graph = concept_graph.graph
    else:
        graph = nx.DiGraph()
        all_concepts = memory.get_all_concepts()
        for concept, data in all_concepts.items():
            graph.add_node(concept, type=data.get("type", "unknown"))

        for fact in memory.get_all_facts():
            src = str(fact.get("from") or "").strip()
            dst = str(fact.get("to") or "").strip()
            if not src or not dst:
                continue
            graph.add_node(src, type=all_concepts.get(src, {}).get("type", "unknown"))
            graph.add_node(dst, type=all_concepts.get(dst, {}).get("type", "unknown"))
            weight = max(0.1, min(1.0, float(fact.get("weight", 0.7) or 0.7)))
            existing = graph.get_edge_data(src, dst, {})
            graph.add_edge(
                src,
                dst,
                relation=existing.get("relation") or str(fact.get("relation") or "related_to"),
                weight=max(float(existing.get("weight", 0.0) or 0.0), weight),
            )

        for path_record in memory.get_all_paths():
            concepts = [str(item).strip() for item in (path_record.get("path") or []) if str(item).strip()]
            path_weight = _clamp_score(path_record.get("score", 0.8), default=0.8)
            for concept in concepts:
                graph.add_node(concept, type=all_concepts.get(concept, {}).get("type", "unknown"))
            for index in range(len(concepts) - 1):
                src = concepts[index]
                dst = concepts[index + 1]
                existing = graph.get_edge_data(src, dst, {})
                graph.add_edge(
                    src,
                    dst,
                    relation=existing.get("relation") or "memory_path",
                    weight=max(float(existing.get("weight", 0.0) or 0.0), path_weight),
                )

    for row in supervision_rows or []:
        normalized = normalize_supervision_row(row)
        for concept in normalized.get("concepts") or []:
            graph.add_node(concept, type=graph.nodes.get(concept, {}).get("type", "proposition"))
        for path in normalized.get("forward_paths") or []:
            for concept in path:
                graph.add_node(concept, type=graph.nodes.get(concept, {}).get("type", "proposition"))
            for index in range(len(path) - 1):
                src = path[index]
                dst = path[index + 1]
                existing = graph.get_edge_data(src, dst, {})
                graph.add_edge(
                    src,
                    dst,
                    relation=existing.get("relation") or "supports",
                    weight=max(float(existing.get("weight", 0.0) or 0.0), _clamp_score(normalized.get("score", 0.8))),
                )
        for fact in normalized.get("facts") or []:
            src = fact["from"]
            dst = fact["to"]
            graph.add_node(src, type=graph.nodes.get(src, {}).get("type", "proposition"))
            graph.add_node(dst, type=graph.nodes.get(dst, {}).get("type", "proposition"))
            existing = graph.get_edge_data(src, dst, {})
            graph.add_edge(
                src,
                dst,
                relation=existing.get("relation") or fact.get("relation", "supports"),
                weight=max(float(existing.get("weight", 0.0) or 0.0), _clamp_score(fact.get("weight", 0.8))),
            )
    return graph


def _parse_zero_shot_spec(spec: str) -> tuple[str, str]:
    normalized = (spec or "").strip().lower()
    if not normalized or normalized in {"none", "off", "false"}:
        return "none", ""
    if ":" in normalized:
        kind, value = normalized.split(":", 1)
        return kind.strip(), value.strip()
    return "concept", normalized


def split_policy_records(
    records: Sequence[PolicyStepRecord],
    *,
    train_ratio: float,
    zero_shot_split: str = "none",
) -> dict[str, list[PolicyStepRecord]]:
    split_map = {"train": [], "val": [], "zero_shot": []}
    kind, value = _parse_zero_shot_spec(zero_shot_split)
    holdout_concepts: set[str] = set()
    holdout_dataset = ""
    if kind == "concept":
        try:
            holdout_ratio = max(0.01, min(0.9, float(value or 0.1)))
        except Exception:
            holdout_ratio = 0.1
        all_concepts = sorted({concept for record in records for concept in record.all_concepts()})
        holdout_concepts = {
            concept.casefold()
            for concept in all_concepts
            if _stable_score(f"zs::{concept.casefold()}") >= (1.0 - holdout_ratio)
        }
    elif kind == "dataset":
        holdout_dataset = value.casefold()

    regular_records: list[PolicyStepRecord] = []
    for record in records:
        if holdout_dataset and record.source_dataset.casefold() == holdout_dataset:
            split_map["zero_shot"].append(record)
            continue
        if holdout_concepts and any(concept.casefold() in holdout_concepts for concept in record.all_concepts()):
            split_map["zero_shot"].append(record)
            continue
        regular_records.append(record)

    boundary = max(0.0, min(1.0, float(train_ratio)))
    for record in regular_records:
        if _stable_score(record.sample_id) < boundary:
            split_map["train"].append(record)
        else:
            split_map["val"].append(record)
    return split_map


class TriMazeNeuralTrainer:
    def __init__(self, graph: nx.DiGraph, memory: ConceptMemory, *, device: str = "auto", seed: int = 42):
        self.graph = graph
        self.memory = memory
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        self.engine = TriMazeEngine(
            self.graph,
            concept_memory=self.memory,
            multimodal_generator=object(),
            policy_enabled=False,
            policy_rollout="off",
        )

    def _resolve_device(self, device_arg: str) -> torch.device:
        if device_arg != "auto":
            return torch.device(device_arg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _memory_forward_paths(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.memory.get_all_paths()
            if str(item.get("mode") or "forward").strip().lower() == "forward"
        ]

    def _record_for_step(
        self,
        *,
        sample_id: str,
        source_kind: str,
        source_dataset: str,
        query_type: str,
        mode: str,
        concepts: Sequence[str],
        step_index: int,
        weight: float,
        source_score: float = 0.0,
    ) -> PolicyStepRecord | None:
        if step_index >= len(concepts) - 1:
            return None
        current = concepts[step_index]
        target = concepts[step_index + 1]
        current_node = self.engine.nodes.get(current)
        if current_node is None:
            return None
        candidate_edges = list(current_node.connections)
        if not candidate_edges:
            return None
        target_index = next(
            (index for index, edge in enumerate(candidate_edges) if edge.to_node.concept == target),
            None,
        )
        if target_index is None:
            return None
        path_stub = build_path_stub(self.engine, concepts[: step_index + 1])
        visited = set(concepts[: step_index + 1])
        return build_runtime_step_record(
            engine=self.engine,
            current_node=current_node,
            candidate_edges=candidate_edges,
            path=path_stub,
            visited=visited,
            mode=mode,
            target_index=target_index,
            sample_id=sample_id,
            source_kind=source_kind,
            source_dataset=source_dataset,
            query_type=query_type,
            task_key=f"{source_dataset}|{query_type}|{mode}",
            weight=weight,
            source_score=source_score,
        )

    def iter_memory_records(self) -> Iterable[PolicyStepRecord]:
        for path_index, path_record in enumerate(self._memory_forward_paths()):
            concepts = [str(item).strip() for item in (path_record.get("path") or []) if str(item).strip()]
            if len(concepts) < 2:
                continue
            weight = _weight_from_path(path_record)
            score = _clamp_score(path_record.get("score", 0.8), default=0.8)
            for step_index in range(len(concepts) - 1):
                record = self._record_for_step(
                    sample_id=f"memory::{path_index}::{step_index}",
                    source_kind="memory_path",
                    source_dataset="concept_memory",
                    query_type="memory",
                    mode="forward",
                    concepts=concepts,
                    step_index=step_index,
                    weight=weight,
                    source_score=score,
                )
                if record is not None:
                    yield record

    def iter_supervision_records(self, supervision_rows: Sequence[dict[str, Any]]) -> Iterable[PolicyStepRecord]:
        for row in supervision_rows:
            normalized = normalize_supervision_row(row)
            paths = list(normalized.get("forward_paths") or [])
            if not paths and normalized.get("facts"):
                paths = [[fact["from"], fact["to"]] for fact in normalized["facts"]]
            for path_index, path in enumerate(paths):
                concepts = [str(item).strip() for item in path if str(item).strip()]
                if len(concepts) < 2:
                    continue
                for step_index in range(len(concepts) - 1):
                    record = self._record_for_step(
                        sample_id=f"supervision::{normalized['sample_id']}::{path_index}::{step_index}",
                        source_kind="tri_maze_supervision",
                        source_dataset=normalized["source_dataset"],
                        query_type=normalized["query_type"],
                        mode="forward",
                        concepts=concepts,
                        step_index=step_index,
                        weight=_clamp_score(normalized.get("score", 0.8), default=0.8),
                        source_score=_clamp_score(normalized.get("score", 0.8), default=0.8),
                    )
                    if record is not None:
                        yield record

    def materialize_records(
        self,
        *,
        supervision_rows: Sequence[dict[str, Any]] | None = None,
        extra_record_files: Sequence[str] | None = None,
    ) -> tuple[list[PolicyStepRecord], dict[str, Any]]:
        records = list(self.iter_memory_records())
        memory_count = len(records)
        supervision_records = list(self.iter_supervision_records(supervision_rows or []))
        records.extend(supervision_records)
        direct_records: list[PolicyStepRecord] = []
        for file_path in extra_record_files or []:
            direct_records.extend(load_policy_records(file_path))
        records.extend(direct_records)

        summary = {
            "record_count": len(records),
            "memory_record_count": memory_count,
            "supervision_record_count": len(supervision_records),
            "direct_record_count": len(direct_records),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "device": str(self.device),
            "feature_schema": {"version": "tri_maze_policy_v2.0"},
        }
        return records, summary

    def prepare_summary(
        self,
        *,
        supervision_rows: Sequence[dict[str, Any]] | None = None,
        extra_record_files: Sequence[str] | None = None,
        train_ratio: float = 0.9,
        zero_shot_split: str = "none",
        cache_dir: str = "",
    ) -> dict[str, Any]:
        records, summary = self.materialize_records(
            supervision_rows=supervision_rows,
            extra_record_files=extra_record_files,
        )
        split_map = split_policy_records(records, train_ratio=train_ratio, zero_shot_split=zero_shot_split)
        vocabulary = PolicyVocabulary.build(split_map["train"] or records)
        summary["splits"] = {key: len(value) for key, value in split_map.items()}
        summary["vocabulary"] = {
            "concepts": vocabulary.concept_vocab_size,
            "relations": vocabulary.relation_vocab_size,
            "sources": vocabulary.source_vocab_size,
            "query_types": vocabulary.query_type_vocab_size,
            "modes": vocabulary.mode_vocab_size,
            "tasks": vocabulary.task_vocab_size,
            "hash_bucket_size": vocabulary.hash_bucket_size,
        }
        if supervision_rows:
            summary["supervision"] = summarize_supervision_rows(supervision_rows)
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            serialize_policy_records(cache_path / "all.jsonl", records)
            serialize_policy_records(cache_path / "train.jsonl", split_map["train"])
            serialize_policy_records(cache_path / "val.jsonl", split_map["val"])
            serialize_policy_records(cache_path / "zero_shot.jsonl", split_map["zero_shot"])
            (cache_path / "manifest.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (cache_path / "vocabulary.json").write_text(
                json.dumps(vocabulary.to_metadata(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary

    def _build_loader(
        self,
        records: Sequence[PolicyStepRecord],
        *,
        vocabulary: PolicyVocabulary,
        config: TrainingConfig,
        train: bool,
    ) -> DataLoader:
        dataset = PolicyStepDataset(records, vocabulary=vocabulary, history_size=config.history_size)
        if train and config.episodic:
            return DataLoader(
                dataset,
                batch_sampler=EpisodicBatchSampler(records, batch_size=config.batch_size, seed=self.seed),
                num_workers=config.num_workers,
                collate_fn=policy_collate_fn,
            )
        sampler = None
        shuffle = bool(train)
        if train and config.domain_balance:
            weights = build_domain_sampling_weights(records)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=policy_collate_fn,
            pin_memory=self.device.type == "cuda",
        )

    def _build_policy(self, *, vocabulary: PolicyVocabulary, config: TrainingConfig) -> EdgePolicy:
        model_config = PolicyModelConfig(
            model_version=config.model_type,
            history_size=config.history_size,
            concept_embedding_dim=config.embedding_dim,
            relation_embedding_dim=config.relation_embedding_dim,
            domain_embedding_dim=config.domain_embedding_dim,
            trunk_dims=tuple(int(item) for item in config.trunk_dims),
            dropout=config.dropout,
            feature_attention=True,
            multitask=config.multitask,
            domain_adapt=config.domain_adapt,
        ).apply_vocabulary(vocabulary)
        policy = EdgePolicy(
            lr=config.lr,
            temperature=config.temperature,
            branch_factor=config.branch_factor,
            revisit_probability=config.revisit_probability,
            seed=self.seed,
            model_version=config.model_type,
            model_config=model_config,
            vocabulary=vocabulary,
            weight_decay=config.weight_decay,
        )
        policy.set_max_degree(self.engine._max_degree)
        policy.model.to(self.device)
        return policy

    def _compute_losses(
        self,
        policy: EdgePolicy,
        batch,
        outputs: dict[str, torch.Tensor],
        config: TrainingConfig,
    ) -> dict[str, torch.Tensor]:
        ranking_loss = masked_cross_entropy(
            outputs["logits"],
            batch.target_index,
            batch.candidate_mask,
            sample_weights=batch.weights,
        )
        hard_loss = hard_negative_margin_loss(
            outputs["logits"],
            batch.target_index,
            batch.candidate_mask,
        )
        contrastive_loss = candidate_contrastive_loss(
            outputs["contrastive_context"],
            outputs["contrastive_candidates"],
            batch.target_index,
            batch.candidate_mask,
        )
        aux_loss = outputs["logits"].new_tensor(0.0)
        if config.multitask:
            aux_loss = aux_loss + F.cross_entropy(outputs["path_length_logits"], batch.path_length_bucket)
            aux_loss = aux_loss + F.binary_cross_entropy_with_logits(outputs["tunnel_logits"], batch.tunnel_label)
            aux_loss = aux_loss + F.binary_cross_entropy_with_logits(outputs["high_value_logits"], batch.high_value_label)
        domain_loss = outputs["logits"].new_tensor(0.0)
        if config.domain_adapt and "domain_logits" in outputs:
            domain_loss = F.cross_entropy(outputs["domain_logits"], batch.source_ids)
        total_loss = ranking_loss
        total_loss = total_loss + config.hard_negative_weight * hard_loss
        total_loss = total_loss + config.contrastive_weight * contrastive_loss
        total_loss = total_loss + config.aux_weight * aux_loss
        total_loss = total_loss + config.domain_adapt_weight * domain_loss
        return {
            "total": total_loss,
            "ranking": ranking_loss.detach(),
            "hard_negative": hard_loss.detach(),
            "contrastive": contrastive_loss.detach(),
            "aux": aux_loss.detach(),
            "domain": domain_loss.detach(),
        }

    def _metric_hits(self, logits: torch.Tensor, target_index: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
        top1 = torch.topk(logits, k=1, dim=-1).indices
        topk = min(3, logits.shape[-1])
        top3 = torch.topk(logits, k=topk, dim=-1).indices
        target = target_index.unsqueeze(-1)
        return int((top1 == target).any(dim=-1).sum().item()), int((top3 == target).any(dim=-1).sum().item())

    def _run_epoch(
        self,
        policy: EdgePolicy,
        loader: DataLoader,
        *,
        config: TrainingConfig,
        train: bool,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        use_amp = bool(config.amp and self.device.type == "cuda")
        autocast = torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:  # pragma: no cover - older torch fallback
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        start = time.perf_counter()
        samples = 0
        steps = 0
        top1_hits = 0
        top3_hits = 0
        loss_totals = {"total": 0.0, "ranking": 0.0, "hard_negative": 0.0, "contrastive": 0.0, "aux": 0.0, "domain": 0.0}

        if train:
            policy.model.train()
            if policy.optimizer is not None:
                policy.optimizer.zero_grad(set_to_none=True)
        else:
            policy.model.eval()

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(self.device)
            samples += batch.batch_size
            steps += 1
            if train:
                with autocast:
                    outputs = policy.model(batch)
                    losses = self._compute_losses(policy, batch, outputs, config)
                    scaled_loss = losses["total"] / max(1, config.grad_accum_steps)
                scaler.scale(scaled_loss).backward()
                if (batch_index + 1) % max(1, config.grad_accum_steps) == 0:
                    scaler.unscale_(policy.optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
                    scaler.step(policy.optimizer)
                    scaler.update()
                    policy.optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    outputs = policy.model(batch)
                    losses = self._compute_losses(policy, batch, outputs, config)

            logits = outputs["logits"]
            hit1, hit3 = self._metric_hits(logits, batch.target_index, batch.candidate_mask)
            top1_hits += hit1
            top3_hits += hit3
            for key in loss_totals:
                loss_totals[key] += float(losses[key].item())

        if train and steps % max(1, config.grad_accum_steps) != 0 and policy.optimizer is not None:
            scaler.unscale_(policy.optimizer)
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
            scaler.step(policy.optimizer)
            scaler.update()
            policy.optimizer.zero_grad(set_to_none=True)

        duration = max(1e-6, time.perf_counter() - start)
        return {
            "loss": loss_totals["total"] / max(1, steps),
            "ranking_loss": loss_totals["ranking"] / max(1, steps),
            "hard_negative_loss": loss_totals["hard_negative"] / max(1, steps),
            "contrastive_loss": loss_totals["contrastive"] / max(1, steps),
            "aux_loss": loss_totals["aux"] / max(1, steps),
            "domain_loss": loss_totals["domain"] / max(1, steps),
            "top1": top1_hits / max(1, samples),
            "top3": top3_hits / max(1, samples),
            "samples": float(samples),
            "steps": float(steps),
            "epoch_time_sec": duration,
            "samples_per_sec": samples / duration,
        }

    def train(
        self,
        *,
        run_dir: str | Path,
        config: TrainingConfig,
        supervision_rows: Sequence[dict[str, Any]] | None = None,
        extra_record_files: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        records, summary = self.materialize_records(
            supervision_rows=supervision_rows,
            extra_record_files=extra_record_files,
        )
        split_map = split_policy_records(records, train_ratio=config.train_ratio, zero_shot_split=config.zero_shot_split)
        train_records = split_map["train"]
        val_records = split_map["val"]
        zero_records = split_map["zero_shot"]
        if not train_records:
            raise RuntimeError("no train records available for Tri-Maze policy training")

        vocabulary = PolicyVocabulary.build(train_records or records)
        policy = self._build_policy(vocabulary=vocabulary, config=config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            policy.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
        if config.resume:
            payload = torch.load(config.resume, map_location="cpu", weights_only=False)
            resume_vocab_matches = False
            if payload.get("format_version") == "tmcra_tri_maze_policy_v2" or (payload.get("config") or {}).get("model_version") == "v2":
                checkpoint_vocab = PolicyVocabulary.from_metadata(payload.get("vocabulary"))
                resume_vocab_matches = checkpoint_vocab.to_metadata() == vocabulary.to_metadata()
            policy.load_checkpoint(
                config.resume,
                load_optimizer=resume_vocab_matches,
                strict=False,
                preserve_vocabulary=not resume_vocab_matches,
            )
            if resume_vocab_matches and payload.get("scheduler_state"):
                scheduler.load_state_dict(payload["scheduler_state"])
            policy.model.to(self.device)

        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        history_path = run_path / "history.jsonl"
        best_val = float("inf")
        best_metrics: dict[str, Any] = {}
        patience_count = 0

        for epoch in range(int(config.epochs)):
            epoch_records = filter_curriculum_records(
                train_records,
                epoch=epoch,
                total_epochs=config.epochs,
                curriculum=config.curriculum,
            )
            train_loader = self._build_loader(epoch_records, vocabulary=vocabulary, config=config, train=True)
            val_loader = self._build_loader(val_records, vocabulary=vocabulary, config=config, train=False) if val_records else None
            zero_loader = self._build_loader(zero_records, vocabulary=vocabulary, config=config, train=False) if zero_records else None

            train_metrics = self._run_epoch(policy, train_loader, config=config, train=True)
            val_metrics = self._run_epoch(policy, val_loader, config=config, train=False) if val_loader is not None else {"loss": 0.0, "top1": 0.0, "top3": 0.0}
            zero_metrics = self._run_epoch(policy, zero_loader, config=config, train=False) if zero_loader is not None else {"loss": 0.0, "top1": 0.0, "top3": 0.0}
            scheduler.step(val_metrics["loss"])

            if self.device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats(self.device)
            else:
                peak_memory_mb = 0.0

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top3": train_metrics["top3"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "zero_shot_loss": zero_metrics["loss"],
                "zero_shot_top1": zero_metrics["top1"],
                "zero_shot_top3": zero_metrics["top3"],
                "samples_per_sec": train_metrics["samples_per_sec"],
                "epoch_time_sec": train_metrics["epoch_time_sec"],
                "peak_memory_mb": peak_memory_mb,
                "lr": float(policy.optimizer.param_groups[0]["lr"]),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            metadata = {
                "epoch": epoch,
                "metrics": row,
                "summary": summary,
                "training_config": config.to_dict(),
                "split_counts": {key: len(value) for key, value in split_map.items()},
            }
            policy.save_checkpoint(
                run_path / "last.pt",
                metadata=metadata,
                scheduler_state=scheduler.state_dict(),
                extra_state={"patience_count": patience_count, "best_val": best_val},
            )
            if val_metrics["loss"] <= best_val:
                best_val = val_metrics["loss"]
                best_metrics = row
                patience_count = 0
                policy.save_checkpoint(
                    run_path / "best.pt",
                    metadata=metadata,
                    scheduler_state=scheduler.state_dict(),
                    extra_state={"patience_count": patience_count, "best_val": best_val},
                )
            else:
                patience_count += 1
                if patience_count >= config.patience:
                    break

        train_summary = {
            **summary,
            "epochs": int(config.epochs),
            "device": str(self.device),
            "best_val_loss": best_val,
            "best_metrics": best_metrics,
            "training_config": config.to_dict(),
        }
        (run_path / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"run_dir": str(run_path), **train_summary}

    def evaluate(
        self,
        *,
        checkpoint: str | Path,
        config: TrainingConfig,
        supervision_rows: Sequence[dict[str, Any]] | None = None,
        extra_record_files: Sequence[str] | None = None,
        split: str = "val",
    ) -> dict[str, Any]:
        records, summary = self.materialize_records(
            supervision_rows=supervision_rows,
            extra_record_files=extra_record_files,
        )
        split_map = split_policy_records(records, train_ratio=config.train_ratio, zero_shot_split=config.zero_shot_split)
        if split == "all":
            eval_records = records
        else:
            eval_records = split_map.get(split, [])
        policy = EdgePolicy(seed=self.seed)
        metadata = policy.load_checkpoint(checkpoint, load_optimizer=False, strict=False)
        policy.set_max_degree(self.engine._max_degree)
        policy.model.to(self.device)
        vocabulary = policy.vocabulary if policy.model_version == "v2" else PolicyVocabulary.build(eval_records or records)
        loader = self._build_loader(eval_records, vocabulary=vocabulary, config=config, train=False)
        metrics = self._run_epoch(policy, loader, config=config, train=False)
        return {
            "checkpoint": str(checkpoint),
            "split": split,
            "device": str(self.device),
            "metrics": metrics,
            "summary": summary,
            "checkpoint_metadata": metadata,
        }

    def benchmark(
        self,
        *,
        config: TrainingConfig,
        supervision_rows: Sequence[dict[str, Any]] | None = None,
        extra_record_files: Sequence[str] | None = None,
        checkpoint: str = "",
        max_batches: int = 20,
    ) -> dict[str, Any]:
        records, summary = self.materialize_records(
            supervision_rows=supervision_rows,
            extra_record_files=extra_record_files,
        )
        split_map = split_policy_records(records, train_ratio=config.train_ratio, zero_shot_split=config.zero_shot_split)
        vocabulary = PolicyVocabulary.build(split_map["train"] or records)
        policy = self._build_policy(vocabulary=vocabulary, config=config)
        if checkpoint:
            policy.load_checkpoint(checkpoint, load_optimizer=False, strict=False)
            policy.model.to(self.device)
        loader = self._build_loader(split_map["train"], vocabulary=vocabulary, config=config, train=True)
        benchmark_metrics = self._run_epoch(policy, loader, config=config, train=not bool(checkpoint), max_batches=max_batches)
        return {"summary": summary, "benchmark": benchmark_metrics, "max_batches": int(max_batches)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Tri-Maze forward edge policy offline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_data_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--memory-file", default=DEFAULT_MEMORY_FILE)
        subparser.add_argument("--graph-json", default="")
        subparser.add_argument("--supervision-file", action="append", default=[])
        subparser.add_argument("--supervision-glob", action="append", default=[])
        subparser.add_argument("--record-file", action="append", default=[])
        subparser.add_argument("--train-ratio", type=float, default=0.9)
        subparser.add_argument("--zero-shot-split", default="none")
        subparser.add_argument("--cache-dir", default="")
        subparser.add_argument("--device", default="auto")
        subparser.add_argument("--seed", type=int, default=42)

    def add_train_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--epochs", type=int, default=20)
        subparser.add_argument("--batch-size", type=int, default=32)
        subparser.add_argument("--num-workers", type=int, default=0)
        subparser.add_argument("--lr", type=float, default=1e-3)
        subparser.add_argument("--weight-decay", type=float, default=1e-4)
        subparser.add_argument("--temperature", type=float, default=1.0)
        subparser.add_argument("--branch-factor", type=int, default=2)
        subparser.add_argument("--revisit-probability", type=float, default=0.2)
        subparser.add_argument("--patience", type=int, default=8)
        subparser.add_argument("--grad-clip", type=float, default=1.0)
        subparser.add_argument("--grad-accum-steps", type=int, default=1)
        subparser.add_argument("--amp", action="store_true")
        subparser.add_argument("--model-type", default="v2")
        subparser.add_argument("--embedding-dim", type=int, default=64)
        subparser.add_argument("--relation-embedding-dim", type=int, default=16)
        subparser.add_argument("--domain-embedding-dim", type=int, default=8)
        subparser.add_argument("--trunk-dims", default="128,256,128")
        subparser.add_argument("--hidden-dim", type=int, default=0)
        subparser.add_argument("--dropout", type=float, default=0.1)
        subparser.add_argument("--history-size", type=int, default=4)
        subparser.add_argument("--curriculum", action="store_true")
        subparser.add_argument("--episodic", action="store_true")
        subparser.add_argument("--no-domain-balance", action="store_true")
        subparser.add_argument("--no-multitask", action="store_true")
        subparser.add_argument("--contrastive-weight", type=float, default=0.08)
        subparser.add_argument("--hard-negative-weight", type=float, default=0.12)
        subparser.add_argument("--aux-weight", type=float, default=0.15)
        subparser.add_argument("--domain-adapt", action="store_true")
        subparser.add_argument("--domain-adapt-weight", type=float, default=0.03)
        subparser.add_argument("--resume", default="")

    prepare = subparsers.add_parser("prepare")
    add_shared_data_args(prepare)
    prepare.add_argument("--output", default="")

    train = subparsers.add_parser("train")
    add_shared_data_args(train)
    add_train_args(train)
    train.add_argument("--run-dir", default="data/tri_maze_policy/runs/latest")

    evaluate = subparsers.add_parser("evaluate")
    add_shared_data_args(evaluate)
    add_train_args(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["train", "val", "zero_shot", "all"], default="val")
    evaluate.add_argument("--output", default="")

    benchmark = subparsers.add_parser("benchmark")
    add_shared_data_args(benchmark)
    add_train_args(benchmark)
    benchmark.add_argument("--checkpoint", default="")
    benchmark.add_argument("--max-batches", type=int, default=20)
    benchmark.add_argument("--output", default="")

    return parser


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    trunk_dims = tuple(int(item.strip()) for item in str(args.trunk_dims).split(",") if item.strip())
    if args.hidden_dim and not trunk_dims:
        trunk_dims = (args.hidden_dim, args.hidden_dim * 2, args.hidden_dim)
    if not trunk_dims:
        trunk_dims = (128, 256, 128)
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        branch_factor=args.branch_factor,
        revisit_probability=args.revisit_probability,
        train_ratio=args.train_ratio,
        patience=args.patience,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        amp=bool(args.amp),
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        relation_embedding_dim=args.relation_embedding_dim,
        domain_embedding_dim=args.domain_embedding_dim,
        trunk_dims=trunk_dims,
        dropout=args.dropout,
        history_size=args.history_size,
        curriculum=CurriculumConfig(enabled=bool(args.curriculum)),
        domain_balance=not bool(args.no_domain_balance),
        episodic=bool(args.episodic),
        multitask=not bool(args.no_multitask),
        contrastive_weight=args.contrastive_weight,
        hard_negative_weight=args.hard_negative_weight,
        aux_weight=args.aux_weight,
        domain_adapt=bool(args.domain_adapt),
        domain_adapt_weight=args.domain_adapt_weight,
        zero_shot_split=args.zero_shot_split,
        cache_dir=args.cache_dir,
        resume=args.resume,
    )


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    memory_file = Path(args.memory_file)
    if not memory_file.is_absolute():
        memory_file = repo_root / memory_file
    memory = ConceptMemory(memory_file=str(memory_file))
    supervision_files = discover_supervision_files(
        repo_root=repo_root,
        files=getattr(args, "supervision_file", []),
        globs=getattr(args, "supervision_glob", []),
    )
    supervision_rows = load_all_supervision_rows(supervision_files)
    graph_json = getattr(args, "graph_json", "") or None
    if graph_json and not Path(graph_json).is_absolute():
        graph_json = str(repo_root / graph_json)
    graph = build_training_graph(memory, supervision_rows=supervision_rows, graph_json_path=graph_json)
    trainer = TriMazeNeuralTrainer(graph, memory, device=args.device, seed=args.seed)

    if args.command == "prepare":
        result = trainer.prepare_summary(
            supervision_rows=supervision_rows,
            extra_record_files=args.record_file,
            train_ratio=args.train_ratio,
            zero_shot_split=args.zero_shot_split,
            cache_dir=args.cache_dir,
        )
    else:
        config = _training_config_from_args(args)
        if args.command == "train":
            run_dir = args.run_dir
            if run_dir.endswith("latest"):
                run_dir = str(Path(run_dir).parent / _timestamp())
            result = trainer.train(
                run_dir=run_dir,
                config=config,
                supervision_rows=supervision_rows,
                extra_record_files=args.record_file,
            )
        elif args.command == "evaluate":
            result = trainer.evaluate(
                checkpoint=args.checkpoint,
                config=config,
                supervision_rows=supervision_rows,
                extra_record_files=args.record_file,
                split=args.split,
            )
        else:
            result = trainer.benchmark(
                config=config,
                supervision_rows=supervision_rows,
                extra_record_files=args.record_file,
                checkpoint=args.checkpoint,
                max_batches=args.max_batches,
            )

    if getattr(args, "output", ""):
        output = Path(args.output)
        if not output.is_absolute():
            output = repo_root / output
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_from_args(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
