"""
概念提取器
从文本中提取机制级概念和因果关系，生成适合 Tri-Maze 推理的高质量 Concept Graph
"""
from __future__ import annotations

import json
import os
from typing import Dict

from loguru import logger
from openai import OpenAI


class ConceptExtractor:
    """机制级概念提取器"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        """初始化概念提取器"""
        self.api_key = (api_key if api_key is not None else os.getenv("API_KEY", "")).strip()
        self.base_url = (base_url if base_url is not None else os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")).strip()
        self.model = (model if model is not None else os.getenv("TMCRA_LLM_MODEL", "deepseek-chat")).strip() or "deepseek-chat"
        self.client = self._build_client()

        self.MAX_CONCEPTS = None
        self.MAX_RELATIONS = 25
        self.system_prompt = """
你是专业的机制级概念提取专家。请从用户输入的文本中提取可用于推理的 Concept Graph，优先生成机制型、因果型的概念和关系，生成适合路径推理的高质量概念图。

# 核心规则：
## 1. 概念提取规则
优先提取以下类型的具体概念，禁止提取抽象无价值概念：
✅ 允许的概念类型：
- structure: 具体结构、部件、实体
- entity: 具体物体、物质、对象
- property: 可测量的属性、特征
- process: 具体过程、动作、机制
- material: 具体材料、物质
- energy: 能量、物理量、信号
- biological structure: 生物结构、组织
- physical mechanism: 物理/化学/生物机制

❌ 禁止提取的抽象概念（绝对不要出现）：
系统、情况、问题、东西、方面、部分、类型、方法、方式、技术、效果、作用、功能

## 2. 关系提取规则
优先提取机制型、因果型关系，禁止模糊关系：
✅ 允许的关系类型：
导致、产生、影响、依赖、组成、包含、调节、转化、传递、驱动、阻碍、催化、生成、连接、控制、供给

❌ 禁止的模糊关系（绝对不要出现）：
相关、有关、涉及、属于、包括、是、有、存在

## 3. 概念链结构要求
尽量构建多层机制因果链，例如：
- 猫 → 有 → 浓密毛发 → 导致 → 隔热 → 维持 → 体温
- 机翼弯曲 → 导致 → 气流速度差 → 产生 → 压力差 → 生成 → 升力
- 光照 → 触发 → 光合作用 → 生成 → 有机物 → 支持 → 植物生长
- 电流 → 流过 → 电阻 → 产生 → 热量 → 升高 → 温度

目标是生成适合路径搜索和推理的链式结构概念图。

## 4. 关系权重规则
权重范围 0-1，根据关系确定性和强度赋值：
- 0.9–1.0：强因果关系、物理定律、确定的机制
- 0.6–0.8：明确的机制关系、组成关系、过程关系
- 0.3–0.5：弱关联、间接影响、不确定关系

## 5. 其他规则
- 避免生成循环关系（A→B 且 B→A），除非是真实的循环过程
- 概念和关系要具体、可推理，不要模糊笼统
- 优先提取能形成长推理链的概念和关系
- 概念数量不设固定上限，尽量完整保留机制链中的关键概念
- 关系优先输出高质量机制关系，受模型上下文窗口约束

# 输出格式要求：
严格输出 JSON 格式，不要任何解释文字、说明、前置或后置内容：
{
  "concepts": [
    {"concept": "概念名称", "type": "structure/entity/property/process/material/energy"},
    {"concept": "概念名称", "type": "structure/entity/property/process/material/energy"}
  ],
  "relations": [
    {"from": "源概念", "to": "目标概念", "relation": "关系描述", "weight": 0.8}
  ]
}
"""

    def _build_client(self):
        if not self.api_key:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def set_api_config(self, api_key: str, base_url: str | None = None, model: str | None = None):
        """设置 API 配置"""
        self.api_key = (api_key or "").strip()
        if base_url is not None and base_url.strip():
            self.base_url = base_url.strip()
        if model is not None and model.strip():
            self.model = model.strip()
        self.client = self._build_client()
        if self.client:
            logger.info("✅ API 配置已更新")
        else:
            logger.info("ℹ️ 概念提取 API 配置已清空")

    def _filter_and_trim_graph(self, result: Dict) -> Dict:
        """过滤和修剪概念图，控制规模，删除低权重关系和孤立节点"""
        concepts = result.get("concepts", [])
        relations = result.get("relations", [])

        if not relations:
            return result

        relations.sort(key=lambda x: x.get("weight", 0), reverse=True)
        if self.MAX_RELATIONS and self.MAX_RELATIONS > 0:
            relations = relations[:self.MAX_RELATIONS]

        used_concepts = set()
        for rel in relations:
            used_concepts.add(rel.get("from", ""))
            used_concepts.add(rel.get("to", ""))

        filtered_concepts = [c for c in concepts if c.get("concept", "") in used_concepts]
        if self.MAX_CONCEPTS and self.MAX_CONCEPTS > 0:
            filtered_concepts = filtered_concepts[:self.MAX_CONCEPTS]

        existing_concept_names = {c.get("concept", "") for c in filtered_concepts}
        for concept_name in used_concepts:
            if concept_name and concept_name not in existing_concept_names and (not self.MAX_CONCEPTS or self.MAX_CONCEPTS <= 0 or len(filtered_concepts) < self.MAX_CONCEPTS):
                filtered_concepts.append({
                    "concept": concept_name,
                    "type": "entity"
                })

        logger.info(
            "✂️ 概念图修剪：{}→{} 个概念，{}→{} 个关系",
            len(concepts),
            len(filtered_concepts),
            len(result.get("relations", [])),
            len(relations),
        )

        return {
            "concepts": filtered_concepts,
            "relations": relations,
        }

    def extract(self, text: str) -> Dict | None:
        """从任意文本中提取机制级概念和因果关系，生成高质量 Concept Graph。"""
        if not self.api_key or not self.client:
            logger.error("❌ 未设置 API Key，请先调用 set_api_config()")
            return None

        logger.info("🔍 提取机制级概念图：{}...", text[:80])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            if "concepts" not in result or "relations" not in result:
                logger.error("❌ API 返回格式错误，缺少 concepts 或 relations 字段")
                return None

            result = self._filter_and_trim_graph(result)
            logger.info("✅ 提取完成：{} 个概念，{} 个关系", len(result["concepts"]), len(result["relations"]))
            logger.debug("提取结果: {}", json.dumps(result, ensure_ascii=False, indent=2))
            return result
        except Exception as exc:
            logger.error("❌ 概念提取失败: {}", exc)
            return None
