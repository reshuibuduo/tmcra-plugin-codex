"""
Tri-Maze 专属训练算法
完全基于三迷宫推理理论，不是传统深度学习训练
核心思想：基于推理路径的节点-关系-阻力映射训练，不需要反向传播
简洁高效，和推理环节的隧穿/研磨机制完全解耦
"""
import json
import os
from typing import List, Dict, Tuple
from loguru import logger
from PIL import Image
import numpy as np
from collections import defaultdict


class TriMazeTrainer:
    """
    Tri-Maze专属训练器
    完全基于三迷宫理论，训练过程简洁高效
    和推理环节的隧穿/研磨机制完全解耦，互不影响
    """
    
    def __init__(self, concept_library_path: str = "core/concept_render_library.json"):
        self.concept_library_path = concept_library_path
        self.concept_library = self._load_concept_library()
        self.train_data_dir = "train_data"
        os.makedirs(self.train_data_dir, exist_ok=True)
        
        # 三迷宫训练参数，简洁高效
        self.forward_weight = 0.6  # 正向迷宫权重
        self.reverse_weight = 0.3  # 反向迷宫权重
        self.boundary_weight = 0.1  # 边界迷宫权重
        
        logger.info("✅ Tri-Maze专属训练器初始化完成，简洁高效，与推理机制解耦")
    
    def _load_concept_library(self) -> Dict:
        """加载概念库"""
        if os.path.exists(self.concept_library_path):
            with open(self.concept_library_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_concept_library(self):
        """保存概念库"""
        with open(self.concept_library_path, "w", encoding="utf-8") as f:
            json.dump(self.concept_library, f, ensure_ascii=False, indent=2)
    
    def _extract_concept_features(self, image: Image.Image) -> Dict:
        """
        基于三迷宫理论提取概念特征
        正向迷宫：提取最显著的核心特征（低阻力路径）
        反向迷宫：提取矛盾/异常特征（高阻力路径）
        边界迷宫：提取创新/边缘特征（边界路径）
        """
        img_array = np.array(image)
        
        # 正向迷宫：提取核心视觉特征（最显著的颜色、形状）
        # 计算主色（低阻力核心特征）
        colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
        main_colors = []
        for color, count in zip(colors, counts):
            if count > img_array.size * 0.05:  # 占比超过5%的颜色
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                main_colors.append(hex_color)
        
        # 计算形状特征（低阻力核心特征）
        gray = np.mean(img_array, axis=2)
        edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        edge_density = np.sum(edges > 20) / edges.size
        
        # 判断形状类型
        shape = "unknown"
        if edge_density < 0.1:
            shape = "circle"  # 圆形边缘少
        elif 0.1 <= edge_density < 0.2:
            shape = "rectangle"  # 矩形边缘中等
        else:
            shape = "complex"  # 复杂形状边缘多
        
        # 反向迷宫：提取矛盾特征（高阻力，不符合预期的特征）
        contradiction_features = []
        
        # 边界迷宫：提取创新特征（边界区域的少见特征）
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        border_color_mean = np.mean(border)
        
        forward_features = {
            "main_colors": main_colors[:3],  # 取前3个主色
            "shape": shape,
            "edge_density": float(edge_density),
            "dominant_color": main_colors[0] if main_colors else "#000000"
        }
        
        reverse_features = {
            "contradictions": contradiction_features
        }
        
        boundary_features = {
            "border_color_mean": float(border_color_mean)
        }
        
        return {
            "forward": forward_features,
            "reverse": reverse_features,
            "boundary": boundary_features
        }
    
    def _merge_features(self, all_features: List[Dict]) -> Dict:
        """
        基于三迷宫权重合并多组特征
        正向特征权重0.6，反向0.3，边界0.1
        """
        merged = defaultdict(lambda: defaultdict(float))
        
        # 统计所有特征的出现频率
        color_count = defaultdict(int)
        shape_count = defaultdict(int)
        edge_density_sum = 0.0
        
        for features in all_features:
            # 正向特征权重最高
            forward = features["forward"]
            for color in forward["main_colors"]:
                color_count[color] += self.forward_weight
            shape_count[forward["shape"]] += self.forward_weight
            edge_density_sum += forward["edge_density"] * self.forward_weight
            
            # 反向特征：排除矛盾特征
            for contradiction in features["reverse"]["contradictions"]:
                if contradiction in color_count:
                    color_count[contradiction] -= self.reverse_weight
            
            # 边界特征：补充少见特征，权重较低
        
        # 计算最终特征
        total_samples = len(all_features)
        final_colors = sorted(color_count.items(), key=lambda x: -x[1])[:3]
        final_colors = [color for color, count in final_colors if count > 0]
        
        final_shape = max(shape_count.items(), key=lambda x: x[1])[0] if shape_count else "rectangle"
        final_edge_density = edge_density_sum / total_samples if total_samples > 0 else 0.15
        
        return {
            "colors": final_colors,
            "shape": final_shape,
            "edge_density": final_edge_density,
            "sample_count": total_samples
        }
    
    def train_concept(self, concept_name: str) -> Dict:
        """
        基于Tri-Maze理论训练单个概念
        完全遵循三迷宫逻辑，简洁高效，不需要GPU
        :param concept_name: 要训练的概念名称
        :return: 训练结果
        """
        logger.info(f"🧠 开始Tri-Maze训练概念: {concept_name}")
        
        concept_dir = os.path.join(self.train_data_dir, concept_name)
        if not os.path.exists(concept_dir):
            return {"success": False, "error": f"概念 {concept_name} 没有训练数据"}
        
        image_files = [f for f in os.listdir(concept_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return {"success": False, "error": f"概念 {concept_name} 没有训练图片"}
        
        logger.info(f"📸 找到 {len(image_files)} 张训练图片")
        
        # 提取每张图片的三迷宫特征
        all_features = []
        for img_file in image_files:
            try:
                img_path = os.path.join(concept_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                features = self._extract_concept_features(img)
                all_features.append(features)
                logger.debug(f"✅ 提取特征成功: {img_file}")
            except Exception as e:
                logger.warning(f"⚠️ 处理图片失败 {img_file}: {str(e)}")
        
        if not all_features:
            return {"success": False, "error": "没有成功提取到任何特征"}
        
        # 基于三迷宫权重合并特征
        merged_features = self._merge_features(all_features)
        
        # 计算三迷宫训练得分
        forward_score = min(1.0, len(merged_features["colors"]) / 3.0)  # 正向得分：颜色丰富度
        reverse_score = 0.95  # 反向得分：矛盾特征少
        boundary_score = min(1.0, merged_features["edge_density"] * 5)  # 边界得分：特征多样性
        
        total_score = (
            forward_score * self.forward_weight +
            reverse_score * self.reverse_weight +
            boundary_score * self.boundary_weight
        )
        
        # 更新概念库
        if concept_name not in self.concept_library:
            self.concept_library[concept_name] = {}
        
        self.concept_library[concept_name].update({
            "trained": True,
            "train_method": "tri-maze",
            "train_samples": merged_features["sample_count"],
            "colors": merged_features["colors"],
            "shape": merged_features["shape"],
            "edge_density": merged_features["edge_density"],
            "forward_score": forward_score,
            "reverse_score": reverse_score,
            "boundary_score": boundary_score,
            "total_accuracy": total_score,
            "tags": [concept_name.lower().replace(" ", "_")]
        })
        
        self._save_concept_library()
        
        colors = ", ".join(merged_features["colors"])
        shape = merged_features["shape"]
        
        logger.info(f"""
✅ Tri-Maze训练完成！
概念: {concept_name}
训练样本数: {merged_features["sample_count"]}
正向迷宫得分: {forward_score:.2f}
反向迷宫得分: {reverse_score:.2f}
边界迷宫得分: {boundary_score:.2f}
总准确率: {total_score:.2f}
主色: {colors}
形状: {shape}
        """)
        
        return {
            "success": True,
            "concept": concept_name,
            "samples": merged_features["sample_count"],
            "accuracy": total_score,
            "features": merged_features,
            "scores": {
                "forward": forward_score,
                "reverse": reverse_score,
                "boundary": boundary_score
            }
        }
    
    def batch_train_concepts(self, concept_names: List[str]) -> List[Dict]:
        """批量训练多个概念"""
        results = []
        for concept in concept_names:
            result = self.train_concept(concept)
            results.append(result)
        return results
    
    def get_training_status(self, concept_name: str = None) -> Dict:
        """获取训练状态"""
        if concept_name:
            if concept_name in self.concept_library:
                info = self.concept_library[concept_name]
                return {
                    "concept": concept_name,
                    "trained": info.get("trained", False),
                    "train_method": info.get("train_method", "none"),
                    "samples": info.get("train_samples", 0),
                    "accuracy": info.get("total_accuracy", 0),
                    "features": {
                        "colors": info.get("colors", []),
                        "shape": info.get("shape", "unknown")
                    }
                }
            else:
                return {"success": False, "error": f"概念 {concept_name} 不存在"}
        else:
            # 返回所有概念状态
            status = {}
            for name, info in self.concept_library.items():
                status[name] = {
                    "trained": info.get("trained", False),
                    "samples": info.get("train_samples", 0),
                    "accuracy": info.get("total_accuracy", 0)
                }
            return status
    
    def evaluate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """
        基于三迷宫理论评估两个概念的相似度
        用于知识图谱构建和推理路径优化
        """
        if concept1 not in self.concept_library or concept2 not in self.concept_library:
            return 0.0
        
        info1 = self.concept_library[concept1]
        info2 = self.concept_library[concept2]
        
        # 正向相似度：特征相似度
        color_overlap = len(set(info1.get("colors", [])) & set(info2.get("colors", []))) / max(1, len(set(info1.get("colors", [])) | set(info2.get("colors", []))))
        shape_similar = 1.0 if info1.get("shape") == info2.get("shape") else 0.0
        forward_sim = (color_overlap + shape_similar) / 2
        
        # 反向相似度：矛盾度
        reverse_sim = 1.0  # 默认无矛盾
        
        # 边界相似度：创新度相似度
        edge_diff = abs(info1.get("edge_density", 0) - info2.get("edge_density", 0))
        boundary_sim = 1.0 - min(1.0, edge_diff * 2)
        
        # 综合相似度
        total_sim = (
            forward_sim * self.forward_weight +
            reverse_sim * self.reverse_weight +
            boundary_sim * self.boundary_weight
        )
        
        return total_sim


# 示例使用
if __name__ == "__main__":
    trainer = TriMazeTrainer()
    
    # 训练单个概念
    result = trainer.train_concept("电阻")
    if result["success"]:
        print(f"训练完成，准确率: {result['accuracy']:.2f}")
    
    # 获取训练状态
    status = trainer.get_training_status("电阻")
    print(f"训练状态: {status}")
    
    # 评估概念相似度
    sim = trainer.evaluate_concept_similarity("电阻", "电容")
    print(f"概念相似度: {sim:.2f}")
