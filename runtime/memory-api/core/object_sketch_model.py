from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class ObjectSketchConfig:
    class_count: int
    style_count: int
    max_strokes: int = 8
    points_per_stroke: int = 32
    class_embed_dim: int = 24
    style_embed_dim: int = 12
    hidden_dim: int = 128
    latent_dim: int = 32

    @property
    def seq_len(self) -> int:
        return self.max_strokes * self.points_per_stroke

    @property
    def point_dim(self) -> int:
        return 3

    @property
    def flat_dim(self) -> int:
        return self.seq_len * self.point_dim


@dataclass(slots=True)
class ObjectSketchV2Config:
    class_count: int
    style_count: int
    scene_count: int
    family_count: int
    max_strokes: int = 12
    points_per_stroke: int = 48
    class_embed_dim: int = 48
    style_embed_dim: int = 24
    scene_embed_dim: int = 16
    family_embed_dim: int = 16
    hidden_dim: int = 256
    latent_dim: int = 96
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1

    @property
    def seq_len(self) -> int:
        return self.max_strokes * self.points_per_stroke

    @property
    def point_dim(self) -> int:
        return 3


@dataclass(slots=True)
class ObjectSketchV3Config:
    class_count: int
    style_count: int
    scene_count: int
    family_count: int
    provenance_count: int = 1
    style_cluster_count: int = 1
    max_strokes: int = 12
    points_per_stroke: int = 48
    class_embed_dim: int = 56
    style_embed_dim: int = 24
    scene_embed_dim: int = 16
    family_embed_dim: int = 16
    provenance_embed_dim: int = 8
    style_cluster_embed_dim: int = 12
    hidden_dim: int = 320
    latent_dim: int = 128
    encoder_layers: int = 2
    dropout: float = 0.1

    @property
    def seq_len(self) -> int:
        return self.max_strokes * self.points_per_stroke

    @property
    def point_dim(self) -> int:
        return 3


if nn is not None:
    class ObjectSketchCVAE(nn.Module):
        def __init__(self, config: ObjectSketchConfig):
            super().__init__()
            self.config = config
            self.class_embed = nn.Embedding(config.class_count, config.class_embed_dim)
            self.style_embed = nn.Embedding(config.style_count, config.style_embed_dim)
            cond_dim = config.class_embed_dim + config.style_embed_dim
            self.encoder = nn.Sequential(
                nn.Linear(config.flat_dim + cond_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.decoder_input = nn.Linear(config.latent_dim + cond_dim, config.hidden_dim)
            self.decoder = nn.GRU(input_size=config.hidden_dim, hidden_size=config.hidden_dim, batch_first=True)
            self.output_head = nn.Linear(config.hidden_dim, config.point_dim)

        def _cond(self, class_ids, style_ids):
            return torch.cat([self.class_embed(class_ids), self.style_embed(style_ids)], dim=-1)

        def encode(self, sequences, class_ids, style_ids):
            batch = sequences.shape[0]
            flattened = sequences.reshape(batch, -1)
            hidden = self.encoder(torch.cat([flattened, self._cond(class_ids, style_ids)], dim=-1))
            return self.mu_head(hidden), self.logvar_head(hidden)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, latent, class_ids, style_ids):
            cond = self._cond(class_ids, style_ids)
            repeated = self.decoder_input(torch.cat([latent, cond], dim=-1)).unsqueeze(1).repeat(1, self.config.seq_len, 1)
            decoded, _ = self.decoder(repeated)
            return self.output_head(decoded)

        def forward(self, sequences, class_ids, style_ids):
            mu, logvar = self.encode(sequences, class_ids, style_ids)
            latent = self.reparameterize(mu, logvar)
            recon = self.decode(latent, class_ids, style_ids)
            return recon, mu, logvar

        def sample(self, class_ids, style_ids, *, latent=None):
            if latent is None:
                latent = torch.randn((class_ids.shape[0], self.config.latent_dim), device=class_ids.device)
            return self.decode(latent, class_ids, style_ids)


    class ObjectSketchV2(nn.Module):
        def __init__(self, config: ObjectSketchV2Config):
            super().__init__()
            self.config = config
            cond_dim = config.class_embed_dim + config.style_embed_dim + config.scene_embed_dim + config.family_embed_dim
            self.class_embed = nn.Embedding(config.class_count, config.class_embed_dim)
            self.style_embed = nn.Embedding(config.style_count, config.style_embed_dim)
            self.scene_embed = nn.Embedding(config.scene_count, config.scene_embed_dim)
            self.family_embed = nn.Embedding(config.family_count, config.family_embed_dim)
            self.input_proj = nn.Linear(config.point_dim, config.hidden_dim)
            self.encoder = nn.GRU(
                input_size=config.hidden_dim + cond_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.encoder_layers,
                batch_first=True,
                dropout=config.dropout if config.encoder_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.context_proj = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + cond_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.position_embed = nn.Embedding(config.seq_len, config.hidden_dim)
            self.decoder_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
            self.decoder_input = nn.Linear(config.latent_dim + cond_dim, config.hidden_dim)
            self.decoder_hidden = nn.Linear(config.latent_dim + cond_dim, config.hidden_dim * config.decoder_layers)
            self.decoder = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.decoder_layers,
                batch_first=True,
                dropout=config.dropout if config.decoder_layers > 1 else 0.0,
            )
            self.output_head = nn.Linear(config.hidden_dim, config.point_dim)
            self.readability_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            self.stroke_count_head = nn.Linear(config.hidden_dim, config.max_strokes + 1)
            self.bbox_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 4),
                nn.Sigmoid(),
            )

        def _cond(self, class_ids, style_ids, scene_ids, family_ids):
            return torch.cat(
                [
                    self.class_embed(class_ids),
                    self.style_embed(style_ids),
                    self.scene_embed(scene_ids),
                    self.family_embed(family_ids),
                ],
                dim=-1,
            )

        def encode(self, sequences, class_ids, style_ids, scene_ids, family_ids, *, mask=None):
            cond = self._cond(class_ids, style_ids, scene_ids, family_ids)
            token = self.input_proj(sequences)
            cond_tokens = cond.unsqueeze(1).expand(-1, sequences.shape[1], -1)
            encoded, _ = self.encoder(torch.cat([token, cond_tokens], dim=-1))
            if mask is None:
                pooled = encoded.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).clamp(0.0, 1.0)
                pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            hidden = self.context_proj(torch.cat([pooled, cond], dim=-1))
            return hidden, self.mu_head(hidden), self.logvar_head(hidden)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, latent, class_ids, style_ids, scene_ids, family_ids):
            cond = self._cond(class_ids, style_ids, scene_ids, family_ids)
            cond_latent = torch.cat([latent, cond], dim=-1)
            batch_size = class_ids.shape[0]
            positions = self.position_embed(
                torch.arange(self.config.seq_len, device=class_ids.device, dtype=torch.long)
            ).unsqueeze(0)
            base = self.decoder_input(cond_latent).unsqueeze(1).expand(batch_size, self.config.seq_len, -1)
            tokens = base + positions + self.decoder_token.expand(batch_size, self.config.seq_len, -1)
            hidden0 = self.decoder_hidden(cond_latent).view(
                self.config.decoder_layers,
                batch_size,
                self.config.hidden_dim,
            ).contiguous()
            decoded, _ = self.decoder(tokens, hidden0)
            raw = self.output_head(decoded)
            points = torch.cat([torch.sigmoid(raw[..., :2]), raw[..., 2:3]], dim=-1)
            summary = decoded.mean(dim=1)
            aux = {
                "readability": self.readability_head(summary).squeeze(-1),
                "stroke_count_logits": self.stroke_count_head(summary),
                "bbox": self.bbox_head(summary),
            }
            return points, aux

        def forward(self, sequences, class_ids, style_ids, scene_ids, family_ids, *, mask=None, **_):
            hidden, mu, logvar = self.encode(
                sequences,
                class_ids,
                style_ids,
                scene_ids,
                family_ids,
                mask=mask,
            )
            latent = self.reparameterize(mu, logvar)
            recon, aux = self.decode(latent, class_ids, style_ids, scene_ids, family_ids)
            aux["context"] = hidden
            aux["latent"] = latent
            return recon, mu, logvar, aux

        def sample(self, class_ids, style_ids, scene_ids, family_ids, *, latent=None, **_):
            if latent is None:
                latent = torch.randn((class_ids.shape[0], self.config.latent_dim), device=class_ids.device)
            recon, aux = self.decode(latent, class_ids, style_ids, scene_ids, family_ids)
            aux["latent"] = latent
            return recon, aux


    class ObjectSketchV3(nn.Module):
        def __init__(self, config: ObjectSketchV3Config):
            super().__init__()
            self.config = config
            cond_dim = (
                config.class_embed_dim
                + config.style_embed_dim
                + config.scene_embed_dim
                + config.family_embed_dim
                + config.provenance_embed_dim
                + config.style_cluster_embed_dim
            )
            self.class_embed = nn.Embedding(config.class_count, config.class_embed_dim)
            self.style_embed = nn.Embedding(config.style_count, config.style_embed_dim)
            self.scene_embed = nn.Embedding(config.scene_count, config.scene_embed_dim)
            self.family_embed = nn.Embedding(config.family_count, config.family_embed_dim)
            self.provenance_embed = nn.Embedding(max(1, config.provenance_count), config.provenance_embed_dim)
            self.style_cluster_embed = nn.Embedding(max(1, config.style_cluster_count), config.style_cluster_embed_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(config.point_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
            )
            self.encoder = nn.GRU(
                input_size=config.hidden_dim + cond_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.encoder_layers,
                batch_first=True,
                dropout=config.dropout if config.encoder_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.context_proj = nn.Sequential(
                nn.Linear(config.hidden_dim * 2 + cond_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.planner = nn.Sequential(
                nn.Linear(config.hidden_dim + cond_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.stroke_count_head = nn.Linear(config.hidden_dim, config.max_strokes + 1)
            self.stroke_length_head = nn.Linear(config.hidden_dim, config.max_strokes)
            self.readability_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            self.naturalness_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            self.style_cluster_head = nn.Linear(config.hidden_dim, max(1, config.style_cluster_count))
            self.bbox_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 4),
                nn.Sigmoid(),
            )
            self.position_embed = nn.Embedding(config.seq_len, config.hidden_dim)
            self.decoder_input = nn.Sequential(
                nn.Linear(config.point_dim + config.latent_dim + cond_dim + config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
            )
            self.decoder_cell = nn.GRUCell(config.hidden_dim, config.hidden_dim)
            self.decoder_hidden = nn.Linear(config.latent_dim + cond_dim + config.hidden_dim, config.hidden_dim)
            self.output_head = nn.Linear(config.hidden_dim, config.point_dim)
            self.sample_context = nn.Sequential(
                nn.Linear(config.latent_dim + cond_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )

        def _cond(self, class_ids, style_ids, scene_ids, family_ids, provenance_ids, style_cluster_ids):
            return torch.cat(
                [
                    self.class_embed(class_ids),
                    self.style_embed(style_ids),
                    self.scene_embed(scene_ids),
                    self.family_embed(family_ids),
                    self.provenance_embed(provenance_ids.clamp_min(0)),
                    self.style_cluster_embed(style_cluster_ids.clamp_min(0)),
                ],
                dim=-1,
            )

        def encode(
            self,
            sequences,
            class_ids,
            style_ids,
            scene_ids,
            family_ids,
            provenance_ids,
            style_cluster_ids,
            *,
            mask=None,
        ):
            cond = self._cond(class_ids, style_ids, scene_ids, family_ids, provenance_ids, style_cluster_ids)
            token = self.input_proj(sequences)
            cond_tokens = cond.unsqueeze(1).expand(-1, sequences.shape[1], -1)
            encoded, _ = self.encoder(torch.cat([token, cond_tokens], dim=-1))
            if mask is None:
                pooled = encoded.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).clamp(0.0, 1.0)
                pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            hidden = self.context_proj(torch.cat([pooled, cond], dim=-1))
            planner_summary = self.planner(torch.cat([hidden, cond], dim=-1))
            return hidden, planner_summary, self.mu_head(hidden), self.logvar_head(hidden), cond

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(
            self,
            latent,
            context,
            cond,
            *,
            teacher_sequence=None,
            teacher_forcing_ratio: float = 1.0,
        ):
            batch_size = latent.shape[0]
            hidden = self.decoder_hidden(torch.cat([latent, cond, context], dim=-1))
            prev_point = torch.zeros((batch_size, self.config.point_dim), device=latent.device, dtype=latent.dtype)
            outputs = []
            for step in range(self.config.seq_len):
                position = self.position_embed(
                    torch.full((batch_size,), step, device=latent.device, dtype=torch.long)
                )
                token = self.decoder_input(torch.cat([prev_point, latent, cond, context], dim=-1)) + position
                hidden = self.decoder_cell(token, hidden)
                raw = self.output_head(hidden)
                point = torch.cat([torch.sigmoid(raw[..., :2]), raw[..., 2:3]], dim=-1)
                outputs.append(point.unsqueeze(1))
                if teacher_sequence is None:
                    prev_point = point.detach()
                    continue
                if teacher_forcing_ratio >= 1.0:
                    prev_point = teacher_sequence[:, step, :]
                    continue
                use_teacher = (
                    torch.rand((batch_size, 1), device=latent.device) < max(0.0, teacher_forcing_ratio)
                ).to(point.dtype)
                prev_point = teacher_sequence[:, step, :] * use_teacher + point.detach() * (1.0 - use_teacher)
            return torch.cat(outputs, dim=1)

        def _build_aux(self, planner_summary, context):
            return {
                "readability": self.readability_head(context).squeeze(-1),
                "stroke_count_logits": self.stroke_count_head(planner_summary),
                "stroke_length_logits": self.stroke_length_head(planner_summary),
                "bbox": self.bbox_head(context),
                "naturalness": self.naturalness_head(context).squeeze(-1),
                "style_cluster_logits": self.style_cluster_head(context),
                "style_embedding": context,
            }

        def forward(
            self,
            sequences,
            class_ids,
            style_ids,
            scene_ids,
            family_ids,
            *,
            provenance_ids=None,
            style_cluster_ids=None,
            mask=None,
            teacher_forcing_ratio: float = 1.0,
        ):
            if provenance_ids is None:
                provenance_ids = torch.zeros_like(class_ids)
            if style_cluster_ids is None:
                style_cluster_ids = torch.zeros_like(class_ids)
            context, planner_summary, mu, logvar, cond = self.encode(
                sequences,
                class_ids,
                style_ids,
                scene_ids,
                family_ids,
                provenance_ids,
                style_cluster_ids,
                mask=mask,
            )
            latent = self.reparameterize(mu, logvar)
            recon = self.decode(
                latent,
                context,
                cond,
                teacher_sequence=sequences,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            aux = self._build_aux(planner_summary, context)
            aux["context"] = context
            aux["latent"] = latent
            return recon, mu, logvar, aux

        def sample(
            self,
            class_ids,
            style_ids,
            scene_ids,
            family_ids,
            *,
            provenance_ids=None,
            style_cluster_ids=None,
            latent=None,
        ):
            if provenance_ids is None:
                provenance_ids = torch.zeros_like(class_ids)
            if style_cluster_ids is None:
                style_cluster_ids = torch.zeros_like(class_ids)
            if latent is None:
                latent = torch.randn((class_ids.shape[0], self.config.latent_dim), device=class_ids.device)
            cond = self._cond(class_ids, style_ids, scene_ids, family_ids, provenance_ids, style_cluster_ids)
            context = self.sample_context(torch.cat([latent, cond], dim=-1))
            recon = self.decode(latent, context, cond, teacher_sequence=None, teacher_forcing_ratio=0.0)
            aux = self._build_aux(context, context)
            aux["context"] = context
            aux["latent"] = latent
            return recon, aux
else:  # pragma: no cover
    class ObjectSketchCVAE:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use ObjectSketchCVAE")


    class ObjectSketchV2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use ObjectSketchV2")


    class ObjectSketchV3:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use ObjectSketchV3")
