"""
真正的迷宫引擎核心
实现 Tri-Maze 概念推理架构 + 隧穿机制 + 自适应概念扩展 + 长期概念记忆 + 回溯式路径研磨机制
"""
import networkx as nx
import math
import os
import random
from typing import List, Dict, Tuple, Set, Any, Literal, Optional
from collections import deque
import asyncio
from loguru import logger
import numpy as np
import json
from .concept_memory import ConceptMemory
from .multimodal_generator import MultimodalGenerator
from .policy_network import EdgePolicy


class MazeNode:
    """迷宫节点 = 概念节点"""
    def __init__(self, concept: str, concept_type: str = "general", context_profile: Dict[str, int] | None = None):
        self.concept = concept
        self.type = concept_type
        self.visited = False
        self.resistance = 0.0  # 节点阻力
        self.connections = []  # 连接的边
        self.expanded = False  # 是否已扩展/研磨过
        self.expand_level = 0  # 研磨层级，0=未研磨，越大越细
        self.expand_time = None
        self.context_profile = context_profile or {}
        self.activation = 0.0  # 扩展时间
    
    def __repr__(self):
        return f"Node({self.concept}, level={self.expand_level}, expanded={self.expanded})"


class MazeEdge:
    """迷宫边 = 概念关系"""
    def __init__(self, from_node: MazeNode, to_node: MazeNode, 
                 relation: str, resistance: float = 0.5):
        self.from_node = from_node
        self.to_node = to_node
        self.relation = relation
        self.resistance = resistance  # 通道阻力，0-1，越小越容易通过
        self.valid = True
        self.is_tunneling = False  # 是否是隧穿生成的边
        self.is_expanded = False  # 是否是扩展生成的边
        self.is_memory = False  # 是否来自记忆
    
    def __repr__(self):
        return f"Edge({self.from_node.concept} → {self.to_node.concept}: {self.relation})"


class MazePath:
    """迷宫路径 = 推理路径"""
    def __init__(self, nodes: List[MazeNode], edges: List[MazeEdge]):
        self.nodes = nodes
        self.edges = edges
        self.total_resistance = sum(e.resistance for e in edges)
        self.valid = True
        self.score_value = 0.0
        self.has_tunneling = any(e.is_tunneling for e in edges)  # 是否包含隧穿
        self.has_expanded = any(e.is_expanded for e in edges)  # 是否包含扩展节点
        self.has_memory = any(e.is_memory for e in edges)  # 是否包含记忆边
        self.failed = False  # 路径是否走不通
    
    @property
    def length(self):
        return len(self.edges)
    
    def add_step(self, node: MazeNode, edge: MazeEdge):
        self.nodes.append(node)
        self.edges.append(edge)
        self.total_resistance += edge.resistance
        if edge.is_tunneling:
            self.has_tunneling = True
        if edge.is_expanded:
            self.has_expanded = True
        if edge.is_memory:
            self.has_memory = True
    
    def copy(self):
        return MazePath(self.nodes.copy(), self.edges.copy())
    
    def get_concept_list(self) -> List[str]:
        """获取路径上的概念列表"""
        return [n.concept for n in self.nodes]
    
    def get_last_n_nodes(self, n: int) -> List[MazeNode]:
        """获取最后n个节点，用于回溯"""
        return self.nodes[-n:] if len(self.nodes) >=n else self.nodes

    def score(self, length_penalty: float = 0.05) -> float:
        return self.total_resistance + length_penalty * self.length


    def __repr__(self):
        path_str = " → ".join([n.concept for n in self.nodes])
        tags = []
        if self.has_tunneling:
            tags.append("TUNNELING")
        if self.has_expanded:
            tags.append("EXPANDED")
        if self.has_memory:
            tags.append("MEMORY")
        if self.failed:
            tags.append("FAILED")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        return f"Path({path_str}, resistance={self.total_resistance:.2f}){tag_str}"


class ReasoningMonitor:
    """推理监控器：实时监控推理状态，决定是否触发回溯研磨"""
    
    def __init__(self, engine):
        self.engine = engine
        self.best_resistance_history = []  # 历史最优路径阻力
        self.stagnation_rounds = 0  # 停滞轮数
        self.max_stagnation_rounds = 3  # 最大停滞轮数，超过触发回溯研磨
        self.min_connectivity_threshold = 2  # 最小连接度阈值
        self.total_nodes_limit = 200  # 总节点数上限
        self.max_expansions_per_round = 3  # 每轮最多研磨节点数
        self.max_grind_level = 3  # 最大研磨层级，避免无限细化
        self.high_resistance_threshold = 0.9  # 高阻力阈值，超过认为路径走不通
        self.backtrack_steps = 1  # 回溯步数，走不通时回退n个节点研磨
    
    def update(self, current_best_resistance: float):
        """更新监控状态"""
        self.best_resistance_history.append(current_best_resistance)
        
        # 检查是否停滞
        if len(self.best_resistance_history) >= 2:
            if abs(current_best_resistance - self.best_resistance_history[-2]) < 0.01:
                self.stagnation_rounds += 1
            else:
                self.stagnation_rounds = 0
    
    def should_grind_path(self, path: MazePath) -> Tuple[bool, Optional[MazeNode], str]:
        """判断是否需要研磨该路径
        :return: (是否需要研磨, 要研磨的节点, 原因)
        """
        def _can_grind(node: MazeNode | None) -> bool:
            return bool(node and node.expand_level < self.max_grind_level and (node.context_profile or {}))

        # 全局限制
        if len(self.engine.nodes) >= self.total_nodes_limit:
            return False, None, "节点总数已达上限"
        
        # 条件1：路径阻力过高，走不通
        if path.score(self.engine.length_penalty) >= self.high_resistance_threshold:
            # 回溯到上一个节点
            backtrack_node = path.get_last_n_nodes(self.backtrack_steps)[0]
            if _can_grind(backtrack_node):
                return True, backtrack_node, "high_resistance"
            return False, None, "节点缺少可研磨上下文"
        
        # 条件2：连续多轮停滞
        if self.stagnation_rounds >= self.max_stagnation_rounds:
            # 选择当前路径中间的节点研磨
            if path.nodes:
                mid_idx = len(path.nodes) // 2
                grind_node = path.nodes[mid_idx]
                if _can_grind(grind_node):
                    return True, grind_node, "stagnation"
            return False, None, "没有适合研磨的节点"
        
        # 条件3：节点连接度过低，没有足够路径可选
        if path.nodes:
            current_node = path.nodes[-1]
            node_degree = len(current_node.connections)
            if node_degree < self.min_connectivity_threshold and _can_grind(current_node):
                return True, current_node, "low_connectivity"
        
        return False, None, "no_need"


class TriMazeEngine:
    """三迷宫引擎核心 + 隧穿机制 + 自适应概念扩展 + 长期记忆 + 回溯式路径研磨"""
    
    def __init__(
        self,
        concept_graph: nx.DiGraph,
        concept_memory: Optional[ConceptMemory] = None,
        multimodal_generator: Optional[MultimodalGenerator] = None,
        policy: EdgePolicy | None = None,
        policy_enabled: bool = True,
        policy_checkpoint_path: str | None = None,
        policy_rollout: Literal["off", "blend"] = "off",
        policy_alpha: float = 0.35,
    ):
        self.graph = concept_graph
        self.memory = concept_memory or ConceptMemory()  # ??????
        self.multimodal_generator = multimodal_generator or MultimodalGenerator(None)  # ??????
        self.nodes: Dict[str, MazeNode] = {}
        self.edges: List[MazeEdge] = []
        self._build_maze()
        self._max_degree = max((len(n.connections) for n in self.nodes.values()), default=1)
        
        # 迷宫配置
        self.max_exploration_steps = 80
        self.max_paths = None
        self.resistance_threshold = 0.8  # 高阻力阈值
        self.exploration_rate = 0.3  # ???????????????
        self.length_penalty = 0.05  # ??????  # 探索率，越大越喜欢尝试未知路径
        
        # 隧穿配置
        self.tunneling_enabled = True
        self.tunneling_probability = 0.2  # 隧穿触发概率
        self.tunneling_max_distance = 5  # 隧穿最大跳跃距离（节点数）
        self.tunneling_min_resistance = 0.3  # 隧穿路径最小阻力
        self.tunneling_validation_enabled = True  # 隧穿路径双重验证
        
        # 路径研磨配置
        self.grinding_enabled = True
        self.reasoning_monitor = ReasoningMonitor(self)

        # Policy network (optional)
        self.policy = policy or EdgePolicy()
        self.policy_enabled = bool(policy_enabled) and self.policy.enabled
        self.policy_branch_factor = 2
        self.policy_rollout: Literal["off", "blend"] = "blend" if policy_rollout == "blend" else "off"
        self.policy_alpha = max(0.0, min(1.0, float(policy_alpha)))
        env_policy_checkpoint = os.getenv("TMCRA_POLICY_CHECKPOINT_PATH", "").strip()
        self.policy_checkpoint_path = (policy_checkpoint_path or env_policy_checkpoint or "").strip() or None
        self.policy_loaded = False
        self.policy_metadata: Dict[str, Any] = {}
        if self.policy_enabled:
            self.policy.set_max_degree(self._max_degree)
            if self.policy_checkpoint_path:
                try:
                    self.policy_metadata = self.policy.load_checkpoint(self.policy_checkpoint_path, load_optimizer=False)
                    self.policy.set_max_degree(self._max_degree)
                    self.policy_loaded = True
                    logger.info("✅ Tri-Maze policy checkpoint loaded: {}", self.policy_checkpoint_path)
                except Exception as exc:
                    logger.warning("Policy checkpoint load failed; fallback to heuristic mode: {}", exc)
                    self.policy_loaded = False
                    self.policy_metadata = {}
            elif policy is not None and self.policy_rollout == "blend":
                self.policy_loaded = True
        if not self.policy_enabled or not self.policy_loaded or self.policy_rollout != "blend":
            self.policy_rollout = "off"
        logger.info("Tri-Maze policy rollout: {}", self.policy_rollout)
        
        logger.info("✅ 三迷宫引擎 + 隧穿 + 回溯研磨 + 长期记忆 初始化完成")
    
    def _build_maze(self):
        """从概念图构建迷宫"""
        # 创建节点
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            self.nodes[node] = MazeNode(
                concept=node,
                concept_type=node_data.get("type", "general"),
                context_profile=node_data.get("context_profile", {}),
            )
        
        # 创建边
        for u, v, data in self.graph.edges(data=True):
            # 应用记忆强化
            reinforcement = self.memory.get_edge_reinforcement(u, v)
            source_kind = str(data.get("source_kind", "")).strip().lower()
            if "resistance" in data:
                try:
                    base_resistance = max(0.0, min(1.0, float(data.get("resistance", 0.5))))
                except Exception:
                    base_resistance = 0.5
            else:
                try:
                    relation_weight = max(0.0, min(1.0, float(data.get("weight", 0.5))))
                except Exception:
                    relation_weight = 0.5
                base_resistance = 1.0 - relation_weight
            final_resistance = max(0.1, base_resistance - reinforcement)  # 强化后阻力降低
            
            edge = MazeEdge(
                from_node=self.nodes[u],
                to_node=self.nodes[v],
                relation=data.get("relation", "related_to"),
                resistance=final_resistance
            )
            if reinforcement > 0:
                edge.is_memory = True  # 标记为记忆边
                
            edge.source_kind = source_kind
            if source_kind.endswith("memory"):
                edge.is_memory = True
            self.edges.append(edge)
            self.nodes[u].connections.append(edge)
    
    def reset_visits(self):
        """重置所有节点访问状态"""
        for node in self.nodes.values():
            node.visited = False

    def _softmax_scores(self, scores: List[float]) -> np.ndarray:
        if not scores:
            return np.asarray([], dtype=np.float64)
        arr = np.asarray(scores, dtype=np.float64)
        shifted = arr - np.max(arr)
        exp = np.exp(shifted)
        total = float(exp.sum())
        if total <= 0:
            return np.full(len(scores), 1.0 / max(1, len(scores)), dtype=np.float64)
        return exp / total

    def _forward_heuristic_probabilities(self, candidate_edges: List[MazeEdge]) -> np.ndarray:
        scores: List[float] = []
        for edge in candidate_edges:
            resistance = float(edge.resistance)
            if resistance < 0.6:
                score = max(0.05, 1.2 - resistance)
            else:
                score = max(0.01, self.exploration_rate * max(0.05, 1.0 - resistance))
            scores.append(score)
        return self._softmax_scores(scores)

    def _select_forward_edges_blend(
        self,
        current_node: MazeNode,
        candidate_edges: List[MazeEdge],
        current_path: MazePath,
        current_visited: Set[str],
    ) -> List[MazeEdge]:
        if not candidate_edges:
            return []
        k = min(self.policy_branch_factor, len(candidate_edges))
        heuristic_prob = self._forward_heuristic_probabilities(candidate_edges)
        evaluated = self.policy.evaluate_candidates(
            engine=self,
            current_node=current_node,
            candidate_edges=candidate_edges,
            path=current_path,
            visited=current_visited,
            mode="forward",
        )
        if evaluated is None:
            return candidate_edges[:k]
        policy_prob = evaluated["probs"].detach().cpu().numpy().astype(np.float64)
        combined = (1.0 - self.policy_alpha) * heuristic_prob + self.policy_alpha * policy_prob
        if not np.isfinite(combined).all() or float(combined.sum()) <= 0.0:
            combined = heuristic_prob
        top_indices = np.argsort(-combined)[:k]
        return [candidate_edges[int(index)] for index in top_indices]

    def _token_set(self, concept: str) -> Set[str]:
        text = concept.lower().replace("_", " ").replace("-", " ")
        if " " in text:
            return {part for part in text.split() if part}
        return {char for char in text if char.strip()}

    
    def _semantic_similarity(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        left_node = self.nodes.get(left)
        right_node = self.nodes.get(right)
        if left_node and right_node:
            left_profile = left_node.context_profile or {}
            right_profile = right_node.context_profile or {}
            if left_profile and right_profile:
                left_top = {k for k, _ in sorted(left_profile.items(), key=lambda item: -item[1])[:10]}
                right_top = {k for k, _ in sorted(right_profile.items(), key=lambda item: -item[1])[:10]}
                if left_top or right_top:
                    overlap = len(left_top & right_top) / max(1, len(left_top | right_top))
                    return max(0.0, min(1.0, overlap))
        # fallback to token overlap
        overlap = 0.0
        left_tokens = self._token_set(left)
        right_tokens = self._token_set(right)
        if left_tokens or right_tokens:
            overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        return max(0.0, min(1.0, overlap))

    def _try_tunneling(self, current_node: MazeNode, visited: Set[str]) -> Optional[Tuple[MazeNode, MazeEdge]]:
        """尝试隧穿：跳跃到远距离节点
        :return: (目标节点, 隧穿边) 隧穿失败返回 None
        """
        if not self.tunneling_enabled or random.random() > self.tunneling_probability:
            return None
        
        # 找到所有未访问的远距离节点
        all_nodes = list(self.nodes.values())
        candidate_nodes = []
        for n in all_nodes:
            if n.concept not in visited and n != current_node:
                # 计算概念距离（简单实现：路径长度）
                try:
                    path_len = nx.shortest_path_length(self.graph, current_node.concept, n.concept)
                    if path_len >= 2:
                        candidate_nodes.append(n)
                except nx.NetworkXNoPath:
                    # 没有路径，视为远距离节点
                    candidate_nodes.append(n)
        
        if not candidate_nodes:
            return None
        
        # 按语义相似度排序，优先选择语义相关节点
        scored = []
        for node in candidate_nodes:
            score = self._semantic_similarity(current_node.concept, node.concept)
            scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_candidates = scored[: min(6, len(scored))]
        if not top_candidates:
            return None
        weights = [max(0.05, score) for score, _ in top_candidates]
        target_node = random.choices([node for _, node in top_candidates], weights=weights, k=1)[0]
        
        # 生成隧穿边
        tunneling_edge = MazeEdge(
            from_node=current_node,
            to_node=target_node,
            relation=f"隧穿连接[{current_node.concept}→{target_node.concept}]",
            resistance=random.uniform(self.tunneling_min_resistance, 0.7)
        )
        tunneling_edge.is_tunneling = True
        
        logger.info(f"🔌 隧穿触发：{current_node.concept} → {target_node.concept} (阻力: {tunneling_edge.resistance:.2f})")
        return target_node, tunneling_edge
    
    
    def _path_visited_concepts(self, path: MazePath | None) -> Set[str]:
        """Return concepts already used in the current path."""
        if not path:
            return set()
        return {
            node.concept
            for node in getattr(path, "nodes", [])
            if getattr(node, "concept", None)
        }

    def _grind_node_native(self, node: MazeNode, count: int = 4) -> List[Dict]:
        """Native grinding based on local co-occurrence profile."""
        if not self.grinding_enabled:
            return []
        if node.expand_level >= self.reasoning_monitor.max_grind_level:
            logger.warning(f"??? ?? {node.concept} ???????? {node.expand_level}")
            return []
        if node.expanded:
            logger.warning(f"??? ?? {node.concept} ???????")
            return []

        profile = node.context_profile or {}
        if not profile:
            logger.debug(f"skip grinding without context profile: {node.concept}")
            return []

        candidates = sorted(profile.items(), key=lambda item: -item[1])
        if not candidates:
            return []

        max_count = max(1, candidates[0][1])
        sub_concepts = []
        for concept, cnt in candidates[:count]:
            if concept == node.concept:
                continue
            weight = max(0.2, min(0.9, cnt / max_count))
            sub_concepts.append({
                "concept": concept,
                "relation": "refines",
                "weight": weight,
            })

        return self._apply_grind(node, sub_concepts)

    def _apply_grind(self, node: MazeNode, sub_concepts: List[Dict]) -> List[Dict]:
        if not sub_concepts:
            return []

        # ????????
        new_nodes = []
        for sub in sub_concepts:
            sub_concept = sub["concept"]
            relation = sub["relation"]
            weight = sub.get("weight", 0.5)

            if sub_concept not in self.nodes:
                self.graph.add_node(sub_concept, type=node.type)
                new_node = MazeNode(sub_concept, node.type)
                new_node.expand_level = node.expand_level + 1
                self.nodes[sub_concept] = new_node
                new_nodes.append(new_node)

                self.memory.add_concept(
                    sub_concept,
                    node.type,
                    "grinding",
                    importance_score=0.6 + node.expand_level * 0.1,
                )
            else:
                new_node = self.nodes[sub_concept]

            self.graph.add_edge(node.concept, sub_concept, relation=relation, weight=weight)
            new_edge = MazeEdge(
                from_node=node,
                to_node=new_node,
                relation=relation,
                resistance=max(0.1, 1 - weight * 0.8),
            )
            new_edge.is_expanded = True
            self.edges.append(new_edge)
            node.connections.append(new_edge)

        node.expanded = True
        node.expand_level += 1

        self.memory.add_concept(
            node.concept,
            node.type,
            "grinding",
            importance_score=0.7,
        )

        logger.info(f"? ????: {node.concept} -> {[s['concept'] for s in sub_concepts]}")
        return sub_concepts

    def _validate_tunneling_path(self, path: MazePath) -> bool:
        """验证隧穿路径：双重验证机制"""
        if not self.tunneling_validation_enabled or not path.has_tunneling:
            return True
        
        logger.info(f"🔍 验证隧穿路径: {path}")
        
        # 正向验证：路径是否能形成合理解释
        if path.score(self.length_penalty) > self.resistance_threshold * 1.2:
            logger.warning(f"❌ 隧穿路径阻力过高，验证失败")
            return False
        
        # 反向验证：是否存在明显矛盾
        concepts = [n.concept for n in path.nodes]
        contradiction_pairs = [
            ("水", "电"), ("火", "水"), ("高温", "塑料"), 
            ("高压", "低压"), ("开", "关"), ("真", "假")
        ]
        
        for a, b in contradiction_pairs:
            if a in concepts and b in concepts:
                logger.warning(f"❌ 隧穿路径存在矛盾：{a} 和 {b} 共存")
                return False
        
        logger.info(f"✅ 隧穿路径验证通过")
        return True
    
    async def forward_maze_explore(self, start_concept: str, target_concept: Optional[str] = None) -> List[MazePath]:
        """
        正向迷宫：探索低阻力路径（最合理的解释）
        支持路径失败回溯研磨机制：路径走不通时回溯研磨节点，重新探索
        """
        logger.info(f"🔍 正向迷宫探索：从 {start_concept} 出发")
        
        if start_concept not in self.nodes:
            logger.error(f"起点概念 {start_concept} 不存在")
            return []
        
        self.reset_visits()
        start_node = self.nodes[start_concept]
        paths: List[MazePath] = []
        queue = deque()
        
        # 初始化路径
        initial_path = MazePath([start_node], [])
        queue.append(initial_path)
        start_node.visited = True
        
        explored = 0
        best_resistance = float('inf')
        grinded_nodes = 0
        max_grinds = self.reasoning_monitor.max_expansions_per_round
        
        while queue and (not self.max_paths or self.max_paths <= 0 or len(paths) < self.max_paths) and explored < self.max_exploration_steps:
            current_path = queue.popleft()
            current_node = current_path.nodes[-1]
            
            # 更新监控器
            if current_path.score(self.length_penalty) < best_resistance:
                best_resistance = current_path.score(self.length_penalty)
            self.reasoning_monitor.update(best_resistance)
            
            # 检查路径是否走不通，需要回溯研磨
            if self.grinding_enabled and grinded_nodes < max_grinds:
                should_grind, grind_node, reason = self.reasoning_monitor.should_grind_path(current_path)
                if should_grind and grind_node:
                    logger.info(f"🔙 路径走不通，触发回溯研磨")
                    logger.info(f"节点: {grind_node.concept}")
                    logger.info(f"原因: {reason}")
                    
                    # 研磨节点
                    new_concepts = self._grind_node_native(grind_node)
                    if new_concepts:
                        logger.info(f"新概念: {[c['concept'] for c in new_concepts]}")
                        grinded_nodes += 1
                        # 研磨后重新初始化队列，从起点开始探索新路径
                        queue = deque([initial_path])
                        self.reset_visits()
                        continue
            
            # 找到目标（如果有）或路径足够长且阻力合理
            if (target_concept and current_node.concept == target_concept) or (not target_concept and current_path.length >= 2 and current_path.score(self.length_penalty) < self.resistance_threshold):
                # 验证隧穿路径
                if self._validate_tunneling_path(current_path):
                    paths.append(current_path)
                continue
            
            # 尝试隧穿
            current_visited = self._path_visited_concepts(current_path)
            tunnel_result = self._try_tunneling(current_node, current_visited)
            if tunnel_result:
                tunnel_node, tunnel_edge = tunnel_result
                if tunnel_node.concept not in current_visited:
                    new_path = current_path.copy()
                    new_path.add_step(tunnel_node, tunnel_edge)
                    queue.append(new_path)
            
            # 探索所有连接
            if self.policy_enabled:
                candidate_edges = current_node.connections
                if candidate_edges:
                    k = min(self.policy_branch_factor, len(candidate_edges))
                    selected_edges = self.policy.select_edges(
                        engine=self,
                        current_node=current_node,
                        candidate_edges=candidate_edges,
                        path=current_path,
                        visited=current_visited,
                        mode="forward",
                        k=k,
                        deterministic=False,
                    )
                    for edge in selected_edges:
                        next_node = edge.to_node
                        if next_node.concept not in current_visited:
                            new_path = current_path.copy()
                            new_path.add_step(next_node, edge)
                            queue.append(new_path)
            else:
                for edge in current_node.connections:
                    next_node = edge.to_node
                    
                    # 低阻力优先 + 一定概率探索未知
                    if edge.resistance < 0.6 or random.random() < self.exploration_rate:
                        if next_node.concept not in current_visited:
                            new_path = current_path.copy()
                            new_path.add_step(next_node, edge)
                            queue.append(new_path)
            
            explored += 1
        
        # 优先保留显式、低阻力、非隧穿路径，避免“捷径”压过主干因果链
        paths.sort(
            key=lambda p: (
                p.score(self.length_penalty),
                1 if p.has_tunneling else 0,
                1 if p.has_memory else 0,
                p.length,
            )
        )
        
        # 记忆钩子：保存成功路径
        for path in paths:
            path_score = 1.0 - path.score(self.length_penalty)  # 阻力越低分数越高
            self.memory.save_successful_path(path.get_concept_list(), path_score)
            
            # 更新路径上概念的重要性
            for node in path.nodes:
                self.memory.update_concept_importance(node.concept, 0.1)

            # Online policy update (optional)
            if self.policy_enabled:
                self.policy.update_from_path(self, path, mode="forward")
        
        tunnel_count = sum(1 for p in paths if p.has_tunneling)
        expanded_count = sum(1 for p in paths if p.has_expanded)
        memory_count = sum(1 for p in paths if p.has_memory)
        
        logger.info(f"✅ 正向迷宫找到 {len(paths)} 条路径，其中 {tunnel_count} 条包含隧穿，{expanded_count} 条包含研磨节点，{memory_count} 条包含记忆边")
        return paths
    
    async def reverse_maze_explore(self, start_concept: str, target_concept: Optional[str] = None) -> List[MazePath]:
        """
        反向迷宫：探索高阻力路径（找矛盾和反例）
        专门找难走的路，验证是否存在反例，支持隧穿
        """
        logger.info(f"🔍 反向迷宫探索：从 {start_concept} 出发找反例")
        
        if start_concept not in self.nodes:
            logger.error(f"起点概念 {start_concept} 不存在")
            return []
        
        self.reset_visits()
        start_node = self.nodes[start_concept]
        paths: List[MazePath] = []
        stack = []  # DFS 优先探索深路径
        
        initial_path = MazePath([start_node], [])
        stack.append(initial_path)
        
        explored = 0
        while stack and (not self.max_paths or self.max_paths <= 0 or len(paths) < self.max_paths) and explored < self.max_exploration_steps:
            current_path = stack.pop()
            current_node = current_path.nodes[-1]
            current_visited = self._path_visited_concepts(current_path)
            
            # 找到矛盾路径或高阻力路径
            if current_path.score(self.length_penalty) > self.resistance_threshold and current_path.length >= 2:
                # 验证隧穿路径
                if self._validate_tunneling_path(current_path):
                    paths.append(current_path)
                continue
            
            # 尝试隧穿（反向迷宫隧穿概率更高）
            original_tunnel_prob = self.tunneling_probability
            self.tunneling_probability = 0.3
            tunnel_result = self._try_tunneling(current_node, current_visited)
            self.tunneling_probability = original_tunnel_prob
            
            if tunnel_result:
                tunnel_node, tunnel_edge = tunnel_result
                if tunnel_node.concept not in current_visited:
                    new_path = current_path.copy()
                    new_path.add_step(tunnel_node, tunnel_edge)
                    stack.append(new_path)
            
            # 优先探索高阻力边
            candidate_edges = [e for e in current_node.connections if e.resistance > self.resistance_threshold * 0.7]
            if self.policy_enabled and candidate_edges:
                k = min(self.policy_branch_factor, len(candidate_edges))
                selected_edges = self.policy.select_edges(
                    engine=self,
                    current_node=current_node,
                    candidate_edges=candidate_edges,
                    path=current_path,
                    visited=current_visited,
                    mode="reverse",
                    k=k,
                    deterministic=True,
                )
                for edge in selected_edges:
                    next_node = edge.to_node
                    if next_node.concept not in current_visited:
                        new_path = current_path.copy()
                        new_path.add_step(next_node, edge)
                        stack.append(new_path)
            else:
                for edge in sorted(current_node.connections, key=lambda e: -e.resistance):
                    next_node = edge.to_node
                    # 高阻力优先，尽量找矛盾
                    if edge.resistance > self.resistance_threshold * 0.7:
                        if next_node.concept not in current_visited:
                            new_path = current_path.copy()
                            new_path.add_step(next_node, edge)
                            stack.append(new_path)
            
            explored += 1
        
        # 按阻力排序，阻力越大越可能是矛盾
        paths.sort(key=lambda p: -p.score(self.length_penalty))
        tunnel_count = sum(1 for p in paths if p.has_tunneling)
        logger.info(f"✅ 反向迷宫找到 {len(paths)} 条高阻力路径（潜在矛盾），其中 {tunnel_count} 条包含隧穿")
        return paths
    
    async def boundary_maze_explore(self, start_concept: str) -> List[MazePath]:
        """
        边界迷宫：探索未知/低概率区域（创新探索）
        探索迷宫边界，发现新连接，隧穿和概念研磨主要发生在这里
        """
        logger.info(f"🔍 边界迷宫探索：从 {start_concept} 出发找创新连接")
        
        if start_concept not in self.nodes:
            logger.error(f"起点概念 {start_concept} 不存在")
            return []
        
        self.reset_visits()
        start_node = self.nodes[start_concept]
        paths: List[MazePath] = []
        
        # 边界迷宫隧穿概率和研磨概率更高
        original_tunnel_prob = self.tunneling_probability
        self.tunneling_probability = 0.4  # 40% 隧穿概率，鼓励创新
        self.tunneling_validation_enabled = False  # 边界探索暂时不严格验证，后续再验证
        original_grind_prob = self.reasoning_monitor.max_stagnation_rounds
        self.reasoning_monitor.max_stagnation_rounds = 2  # 更容易触发研磨
        
        # 随机游走探索边界
        for _ in range(8):  # 尝试8次随机游走
            current_node = start_node
            path = MazePath([current_node], [])
            visited = set([current_node.concept])
            
            for _ in range(6):  # 最多走6步
                # 优先尝试研磨低连接度节点
                if self.grinding_enabled and len(current_node.connections) < 2 and current_node.expand_level < 2 and (current_node.context_profile or {}):
                    self._grind_node_native(current_node)
                
                # 优先尝试隧穿
                tunnel_result = self._try_tunneling(current_node, visited)
                if tunnel_result:
                    next_node, edge = tunnel_result
                    visited.add(next_node.concept)
                    path.add_step(next_node, edge)
                    current_node = next_node
                    continue
                
                # 否则选未访问的边
                unvisited_edges = [e for e in current_node.connections if e.to_node.concept not in visited]
                if not unvisited_edges:
                    break
                
                # 优先使用策略网络选择（否则随机）
                if self.policy_enabled:
                    selected = self.policy.select_edges(
                        engine=self,
                        current_node=current_node,
                        candidate_edges=unvisited_edges,
                        path=path,
                        visited=visited,
                        mode="boundary",
                        k=1,
                        deterministic=False,
                    )
                    if not selected:
                        break
                    edge = selected[0]
                else:
                    # 随机选边，优先选阻力中等的（边界区域）
                    edge = random.choice(unvisited_edges)
                next_node = edge.to_node
                visited.add(next_node.concept)
                path.add_step(next_node, edge)
                current_node = next_node
            
            if path.length >= 3:
                # 验证隧穿路径
                if self._validate_tunneling_path(path):
                    paths.append(path)
                    
                    # 记忆钩子：保存创新路径
                    path_score = 0.7  # 创新路径基础分
                    self.memory.save_successful_path(path.get_concept_list(), path_score)
                    if self.policy_enabled:
                        self.policy.update_from_path(self, path, mode="boundary")
        
        # 恢复原始配置
        self.tunneling_probability = original_tunnel_prob
        self.tunneling_validation_enabled = True
        self.reasoning_monitor.max_stagnation_rounds = original_grind_prob
        
        tunnel_count = sum(1 for p in paths if p.has_tunneling)
        expanded_count = sum(1 for p in paths if p.has_expanded)
        logger.info(f"✅ 边界迷宫找到 {len(paths)} 条创新路径，其中 {tunnel_count} 条包含隧穿，{expanded_count} 条包含研磨节点")
