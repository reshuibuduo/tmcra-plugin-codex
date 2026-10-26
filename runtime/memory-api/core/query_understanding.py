"""
LLM front-layer query understanding.
Normalizes user queries into structured seeds before graph extraction/search.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from loguru import logger
from openai import OpenAI


class QueryUnderstandingLayer:
    """LLM-based front-layer understanding for TMCRA."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = (api_key if api_key is not None else os.getenv("API_KEY", "")).strip()
        self.base_url = (base_url if base_url is not None else os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")).strip()
        self.model = (model if model is not None else os.getenv("TMCRA_QUERY_MODEL", os.getenv("TMCRA_LLM_MODEL", "deepseek-chat"))).strip() or "deepseek-chat"
        self.max_concepts = None
        self.max_relations = 18
        self.client = self._build_client()
        self.system_prompt = """
你是 TMCRA 的前置理解层。你的任务不是直接回答问题，而是把用户自然语言整理成适合后续图推理系统处理的结构化输入。

目标：
1. 判断问题意图（general / explanation / necessity / how_to / design / code）
2. 把原问题改写成更适合机制推理的规范化问题
3. 提取核心概念、候选关系、关注焦点
4. 尽量避免抽象空泛概念，优先保留可进入知识图谱的实体、结构、过程、属性、材料、能量
5. 输出严格 JSON，不要额外解释

输出格式：
{
  "intent": "explanation",
  "normalized_query": "解释 LED 串联电阻限制电流的机制",
  "focus_concept": "LED",
  "concepts": [
    {"concept": "LED", "type": "entity"},
    {"concept": "电阻", "type": "component"}
  ],
  "relations": [
    {"from": "电阻", "to": "电流", "relation": "限制", "weight": 0.78}
  ],
  "constraints": ["优先机制链", "避免抽象概念"],
  "confidence": 0.86
}

要求：
- concepts 不设固定上限，尽量完整保留关键概念
- relations 最多 18 个
- weight 在 0 到 1 之间
- normalized_query 必须保留用户原始意图，但改写得更适合图推理
- 如果用户问题本身很清楚，也可以基本保持原句
- 如果无法判断，就保守输出，confidence 降低
"""

    def _build_client(self):
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def set_api_config(self, api_key: str, base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = (api_key or "").strip()
        if base_url is not None and base_url.strip():
            self.base_url = base_url.strip()
        if model is not None and model.strip():
            self.model = model.strip()
        self.client = self._build_client()
        if self.client:
            logger.info("✅ 前置理解层 API 配置已更新")
        else:
            logger.info("ℹ️ 前置理解层 API 配置已清空")

    def _clamp_weight(self, value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.5

    def _normalize_concepts(self, concepts: List[Dict]) -> List[Dict]:
        seen = set()
        normalized: List[Dict] = []
        for item in concepts or []:
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept", "")).strip()
            if not concept or concept in seen:
                continue
            seen.add(concept)
            normalized.append({
                "concept": concept,
                "type": str(item.get("type", "general") or "general").strip() or "general",
            })
            if self.max_concepts and self.max_concepts > 0 and len(normalized) >= self.max_concepts:
                break
        return normalized

    def _normalize_relations(self, relations: List[Dict], concept_names: set[str]) -> List[Dict]:
        seen = set()
        normalized: List[Dict] = []
        for item in relations or []:
            if not isinstance(item, dict):
                continue
            src = str(item.get("from", "")).strip()
            dst = str(item.get("to", "")).strip()
            relation = str(item.get("relation", "")).strip()
            if not src or not dst or not relation or src == dst:
                continue
            key = (src, dst, relation)
            if key in seen:
                continue
            seen.add(key)
            if concept_names and (src not in concept_names or dst not in concept_names):
                continue
            normalized.append({
                "from": src,
                "to": dst,
                "relation": relation,
                "weight": self._clamp_weight(item.get("weight", 0.6)),
            })
            if len(normalized) >= self.max_relations:
                break
        return normalized

    def _normalize_result(self, result: Dict, query: str) -> Dict:
        concepts = self._normalize_concepts(result.get("concepts", []))
        concept_names = {item["concept"] for item in concepts}
        relations = self._normalize_relations(result.get("relations", []), concept_names)
        focus = str(result.get("focus_concept", "")).strip()
        if focus and focus not in concept_names:
            focus = ""
        if not focus and concepts:
            focus = concepts[0]["concept"]

        intent = str(result.get("intent", "general") or "general").strip().lower()
        if intent not in {"general", "explanation", "necessity", "how_to", "design", "code"}:
            intent = "general"

        constraints = []
        for item in result.get("constraints", []):
            text = str(item).strip()
            if text:
                constraints.append(text)
        constraints = constraints[:6]

        normalized_query = str(result.get("normalized_query", "")).strip() or query.strip()
        confidence = self._clamp_weight(result.get("confidence", 0.5))

        return {
            "intent": intent,
            "normalized_query": normalized_query,
            "focus_concept": focus,
            "concepts": concepts,
            "relations": relations,
            "constraints": constraints,
            "confidence": confidence,
        }

    def preprocess(self, query: str) -> Dict | None:
        text = query.strip()
        if not text:
            return None
        if not self.available:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            normalized = self._normalize_result(result, text)
            logger.info(
                "✅ 前置理解完成：intent={} concepts={} relations={} confidence={}",
                normalized["intent"],
                len(normalized["concepts"]),
                len(normalized["relations"]),
                normalized["confidence"],
            )
            return normalized
        except Exception as exc:
            logger.warning("前置理解层调用失败: {}", exc)
            return None
