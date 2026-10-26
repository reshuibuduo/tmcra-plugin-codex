"""
概念图存储
基于 NetworkX 实现
"""
import networkx as nx
from typing import Dict, List
from loguru import logger


class ConceptGraph:
    """概念图存储"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        logger.info("✅ 概念图初始化完成")
    
    def add_concept(self, concept: str, concept_type: str = "general"):
        """添加概念"""
        if not self.graph.has_node(concept):
            self.graph.add_node(concept, type=concept_type)
            logger.debug(f"添加概念: {concept} ({concept_type})")
    
    def add_relation(self, from_concept: str, to_concept: str, relation: str, weight: float = 0.5):
        """添加概念之间的关系"""
        if not self.graph.has_node(from_concept):
            self.add_concept(from_concept)
        if not self.graph.has_node(to_concept):
            self.add_concept(to_concept)
        
        self.graph.add_edge(from_concept, to_concept, relation=relation, weight=weight)
        logger.debug(f"添加关系: {from_concept} → {to_concept}: {relation} (权重: {weight})")
    
    def remove_concept(self, concept: str):
        """删除概念"""
        if self.graph.has_node(concept):
            self.graph.remove_node(concept)
            logger.debug(f"删除概念: {concept}")
    
    def remove_relation(self, from_concept: str, to_concept: str):
        """删除关系"""
        if self.graph.has_edge(from_concept, to_concept):
            self.graph.remove_edge(from_concept, to_concept)
            logger.debug(f"删除关系: {from_concept} → {to_concept}")
    
    def get_concepts(self) -> List[str]:
        """获取所有概念"""
        return list(self.graph.nodes)
    
    def get_relations(self) -> List[Dict]:
        """获取所有关系"""
        relations = []
        for u, v, data in self.graph.edges(data=True):
            relations.append({
                "from": u,
                "to": v,
                "relation": data.get("relation", ""),
                "weight": data.get("weight", 0.5)
            })
        return relations
    
    def find_paths(self, start: str, end: str = None, max_depth: int = 5) -> List[List[str]]:
        """查找概念之间的路径"""
        if end:
            # 查找两点之间的所有简单路径
            paths = list(nx.all_simple_paths(self.graph, source=start, target=end, cutoff=max_depth))
        else:
            # 从起点出发的所有路径
            paths = []
            for node in self.graph.nodes:
                if node != start:
                    try:
                        paths.extend(nx.all_simple_paths(self.graph, source=start, target=node, cutoff=max_depth))
                    except nx.NetworkXNoPath:
                        pass
        
        logger.debug(f"找到 {len(paths)} 条路径从 {start} 出发")
        return paths
    
    def clear(self):
        """清空概念图"""
        self.graph.clear()
        logger.debug("概念图已清空")
    
    def get_graph(self) -> nx.DiGraph:
        """获取 NetworkX 图对象"""
        return self.graph
    
    def export_json(self) -> Dict:
        """导出为 JSON 格式"""
        data = {
            "concepts": [],
            "relations": []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            data["concepts"].append({
                "concept": node,
                "type": attrs.get("type", "general")
            })
        
        for u, v, attrs in self.graph.edges(data=True):
            data["relations"].append({
                "from": u,
                "to": v,
                "relation": attrs.get("relation", ""),
                "weight": attrs.get("weight", 0.5)
            })
        
        return data
    
    def import_json(self, data: Dict):
        """从 JSON 导入"""
        self.clear()
        
        for concept in data.get("concepts", []):
            self.add_concept(concept["concept"], concept.get("type", "general"))
        
        for relation in data.get("relations", []):
            self.add_relation(
                relation["from"],
                relation["to"],
                relation["relation"],
                relation.get("weight", 0.5)
            )
        
        logger.info(f"导入了 {len(self.graph.nodes)} 个概念，{len(self.graph.edges)} 个关系")
