"""
Tri-Maze 高级渲染引擎 v2.0
扩展优化版，增加更多高级特效、3D效果、材质系统、粒子系统
"""
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
import random

class AdvancedRendererV2:
    """
    高级渲染引擎v2.0，扩展优化版
    增加3D效果、材质系统、粒子系统、高级滤镜等
    """
    
    def __init__(self, width: int = 1024, height: int = 768, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.layers = []
        self.current_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.current_layer)
        self.bg_color = bg_color + (255,)
        self.light_sources = []  # 光源列表，支持3D光照
    
    def create_layer(self) -> int:
        """创建新图层"""
        self.layers.append(self.current_layer)
        self.current_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.current_layer)
        return len(self.layers)
    
    def merge_layers(self) -> Image.Image:
        """合并所有图层，支持光照计算"""
        final = Image.new("RGBA", (self.width, self.height), self.bg_color)
        
        # 应用全局光照
        if self.light_sources:
            final = self._apply_global_lighting(final)
        
        for layer in self.layers + [self.current_layer]:
            final = Image.alpha_composite(final, layer)
        
        return final.convert("RGB")
    
    # ===== 光照系统 =====
    def add_light_source(self, x: int, y: int, intensity: float = 1.0, 
                        color: Tuple[int, int, int] = (255, 255, 255),
                        radius: int = 300):
        """添加光源，支持3D光照效果"""
        self.light_sources.append({
            "x": x,
            "y": y,
            "intensity": intensity,
            "color": color,
            "radius": radius
        })
    
    def _apply_global_lighting(self, image: Image.Image) -> Image.Image:
        """应用全局光照效果"""
        light_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(light_layer)
        
        for light in self.light_sources:
            # 径向渐变光晕
            for r in range(light["radius"], 0, -10):
                ratio = r / light["radius"]
                alpha = int(200 * light["intensity"] * (1 - ratio))
                if alpha <= 0:
                    break
                color = (
                    int(light["color"][0] * (1 - ratio * 0.7)),
                    int(light["color"][1] * (1 - ratio * 0.7)),
                    int(light["color"][2] * (1 - ratio * 0.7)),
                    alpha
                )
                draw.ellipse(
                    [light["x"] - r, light["y"] - r, 
                     light["x"] + r, light["y"] + r],
                    fill=color
                )
        
        # 屏幕混合模式
        return ImageChops.screen(image, light_layer)
    
    # ===== 3D效果 =====
    def draw_3d_cube(self, x: int, y: int, size: int, 
                    face_colors: List[Tuple[int, int, int, int]] = None,
                    rotation: float = 0.5):
        """绘制3D立方体"""
        if face_colors is None:
            face_colors = [
                (200, 200, 200, 255),  # 前面
                (150, 150, 150, 255),  # 侧面
                (100, 100, 100, 255)   # 顶面
            ]
        
        # 立方体顶点坐标（透视投影）
        z = size * 0.5
        points = [
            # 前面四个点
            (x - size, y - size, z),
            (x + size, y - size, z),
            (x + size, y + size, z),
            (x - size, y + size, z),
            # 后面四个点
            (x - size + size*rotation, y - size - size*rotation, -z),
            (x + size + size*rotation, y - size - size*rotation, -z),
            (x + size + size*rotation, y + size - size*rotation, -z),
            (x - size + size*rotation, y + size - size*rotation, -z),
        ]
        
        # 投影到2D平面
        projected = []
        for px, py, pz in points:
            scale = 1 + (pz / (size * 3))
            projected.append((int(px * scale), int(py * scale)))
        
        # 绘制面
        faces = [
            ([0, 1, 2, 3], face_colors[0]),  # 前面
            ([1, 5, 6, 2], face_colors[1]),  # 右面
            ([0, 4, 7, 3], face_colors[1]),  # 左面
            ([4, 5, 6, 7], face_colors[2]),  # 后面
            ([0, 1, 5, 4], face_colors[2]),  # 顶面
            ([3, 2, 6, 7], face_colors[1]),  # 底面
        ]
        
        # 按z轴排序，远处的面先画
        faces.sort(key=lambda f: sum(points[i][2] for i in f[0])/4)
        
        for face_indices, color in faces:
            face_points = [projected[i] for i in face_indices]
            self.draw.polygon(face_points, fill=color, outline=(50, 50, 50, 255), width=1)
    
    def draw_3d_sphere(self, x: int, y: int, radius: int,
                      color: Tuple[int, int, int, int] = (100, 150, 255, 255),
                      light_pos: Tuple[int, int] = None):
        """绘制3D球体，带光照效果"""
        if light_pos is None:
            light_pos = (x - radius//2, y - radius//2)
        
        # 径向渐变模拟3D效果
        for r in range(radius, 0, -1):
            # 计算光照
            dx = light_pos[0] - x
            dy = light_pos[1] - y
            distance = math.hypot(dx, dy)
            light_ratio = 1 - (r / radius) * 0.7
            
            r_color = int(color[0] * light_ratio)
            g_color = int(color[1] * light_ratio)
            b_color = int(color[2] * light_ratio)
            
            self.draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=(r_color, g_color, b_color, color[3])
            )
        
        # 高光
        highlight_radius = radius // 5
        highlight_x = x - radius//3
        highlight_y = y - radius//3
        self.draw.ellipse(
            [highlight_x - highlight_radius, highlight_y - highlight_radius,
             highlight_x + highlight_radius, highlight_y + highlight_radius],
            fill=(255, 255, 255, 180)
        )
    
    # ===== 材质系统 =====
    def apply_material(self, layer_index: int = None, material_type: str = "glass"):
        """应用材质效果"""
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        
        if material_type == "glass":
            # 玻璃材质：透明+模糊+高光
            blurred = target_layer.filter(ImageFilter.GaussianBlur(2))
            result = Image.blend(target_layer, blurred, 0.3)
            # 添加高光
            highlight = Image.new("RGBA", target_layer.size, (255, 255, 255, 30))
            result = Image.alpha_composite(result, highlight)
        
        elif material_type == "metal":
            # 金属材质：高对比度+反光
            np_img = np.array(target_layer)
            np_img[..., :3] = np.clip(np_img[..., :3] * 1.3, 0, 255)
            result = Image.fromarray(np_img.astype(np.uint8), "RGBA")
            # 添加反光
            edge = ImageOps.expand(target_layer, border=2, fill=(255, 255, 255, 100))
            result.paste(edge, (0, 0), edge)
        
        elif material_type == "wood":
            # 木质材质：木纹纹理+暖色调
            noise = np.random.normal(0, 15, target_layer.size + (3,))
            np_img = np.array(target_layer)
            np_img[..., :3] = np.clip(np_img[..., :3] + noise, 0, 255)
            # 暖色调
            np_img[..., 0] = np.clip(np_img[..., 0] * 1.2, 0, 255)
            np_img[..., 1] = np.clip(np_img[..., 1] * 1.1, 0, 255)
            np_img[..., 2] = np.clip(np_img[..., 2] * 0.8, 0, 255)
            result = Image.fromarray(np_img.astype(np.uint8), "RGBA")
        
        else:  # 默认材质
            result = target_layer
        
        if layer_index is None:
            self.current_layer = result
        else:
            self.layers[layer_index] = result
    
    # ===== 粒子系统 =====
    def draw_particle_system(self, x: int, y: int, count: int = 50,
                           particle_color: Tuple[int, int, int, int] = (255, 200, 100, 200),
                           spread: int = 100,
                           particle_size: Tuple[int, int] = (2, 6)):
        """绘制粒子系统，用于火焰、星光、烟雾等效果"""
        for _ in range(count):
            # 随机位置
            offset_x = random.randint(-spread, spread)
            offset_y = random.randint(-spread, spread)
            px = x + offset_x
            py = y + offset_y
            
            # 随机大小
            size = random.randint(particle_size[0], particle_size[1])
            
            # 随机透明度
            alpha = random.randint(100, particle_color[3])
            color = (particle_color[0], particle_color[1], particle_color[2], alpha)
            
            # 绘制粒子
            self.draw_circle(px, py, size, fill=color)
    
    # ===== 高级滤镜 =====
    def apply_vignette(self, intensity: float = 0.5):
        """应用暗角效果"""
        vignette = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(vignette)
        
        for r in range(max(self.width, self.height), 0, -10):
            brightness = int(255 * (1 - intensity * (1 - r / max(self.width, self.height))))
            draw.ellipse(
                [self.width//2 - r, self.height//2 - r,
                 self.width//2 + r, self.height//2 + r],
                fill=brightness
            )
        
        # 应用暗角到所有图层
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.putalpha(vignette)
            self.layers[i] = layer
        
        current_vignette = Image.new("RGBA", (self.width, self.height))
        current_vignette.putalpha(vignette)
        self.current_layer = Image.alpha_composite(self.current_layer, current_vignette)
    
    def apply_color_grading(self, brightness: float = 1.0, contrast: float = 1.0,
                          saturation: float = 1.0, temperature: float = 0.0):
        """应用色彩分级，调整亮度、对比度、饱和度、色温"""
        np_img = np.array(self.current_layer)
        
        # 亮度调整
        np_img[..., :3] = np.clip(np_img[..., :3] * brightness, 0, 255)
        
        # 对比度调整
        mean = np.mean(np_img[..., :3])
        np_img[..., :3] = np.clip((np_img[..., :3] - mean) * contrast + mean, 0, 255)
        
        # 色温调整
        if temperature > 0:  # 暖色调
            np_img[..., 0] = np.clip(np_img[..., 0] * (1 + temperature), 0, 255)
            np_img[..., 2] = np.clip(np_img[..., 2] * (1 - temperature), 0, 255)
        else:  # 冷色调
            np_img[..., 0] = np.clip(np_img[..., 0] * (1 + temperature), 0, 255)
            np_img[..., 2] = np.clip(np_img[..., 2] * (1 - temperature), 0, 255)
        
        self.current_layer = Image.fromarray(np_img.astype(np.uint8), "RGBA")
    
    def apply_bloom(self, threshold: int = 200, blur_radius: float = 10.0):
        """应用 Bloom 发光效果，亮部溢出"""
        # 提取亮部
        np_img = np.array(self.current_layer)
        bright_mask = (np_img[..., 0] > threshold) & (np_img[..., 1] > threshold) & (np_img[..., 2] > threshold)
        bright_parts = np.zeros_like(np_img)
        bright_parts[bright_mask] = np_img[bright_mask]
        
        bright_layer = Image.fromarray(bright_parts.astype(np.uint8), "RGBA")
        blurred_bright = bright_layer.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # 混合发光效果
        self.current_layer = Image.alpha_composite(self.current_layer, blurred_bright)
    
    # ===== 基础形状（继承v1版并增强） =====
    def draw_rectangle(self, x: int, y: int, width: int, height: int, 
                      fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
                      outline: Tuple[int, int, int, int] = None,
                      stroke_width: int = 1,
                      corner_radius: int = 0):
        """绘制矩形，支持圆角"""
        if corner_radius == 0:
            self.draw.rectangle([x, y, x + width, y + height], fill=fill, outline=outline, width=stroke_width)
            return
        
        # 圆角矩形
        radius = corner_radius
        self.draw.ellipse([x, y, x + 2*radius, y + 2*radius], fill=fill, outline=outline, width=stroke_width)
        self.draw.ellipse([x + width - 2*radius, y, x + width, y + 2*radius], fill=fill, outline=outline, width=stroke_width)
        self.draw.ellipse([x, y + height - 2*radius, x + 2*radius, y + height], fill=fill, outline=outline, width=stroke_width)
        self.draw.ellipse([x + width - 2*radius, y + height - 2*radius, x + width, y + height], fill=fill, outline=outline, width=stroke_width)
        
        self.draw.rectangle([x + radius, y, x + width - radius, y + height], fill=fill, outline=None)
        self.draw.rectangle([x, y + radius, x + width, y + height - radius], fill=fill, outline=None)
        
        if outline:
            self.draw.line([x + radius, y, x + width - radius, y], fill=outline, width=stroke_width)
            self.draw.line([x + radius, y + height, x + width - radius, y + height], fill=outline, width=stroke_width)
            self.draw.line([x, y + radius, x, y + height - radius], fill=outline, width=stroke_width)
            self.draw.line([x + width, y + radius, x + width, y + height - radius], fill=outline, width=stroke_width)
    
    def draw_circle(self, x: int, y: int, radius: int,
                   fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
                   outline: Tuple[int, int, int, int] = None,
                   stroke_width: int = 1):
        """绘制圆形"""
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                         fill=fill, outline=outline, width=stroke_width)
    
    def draw_linear_gradient(self, x: int, y: int, width: int, height: int,
                           start_color: Tuple[int, int, int, int],
                           end_color: Tuple[int, int, int, int],
                           direction: str = "vertical"):
        """绘制线性渐变"""
        gradient = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(gradient)
        
        if direction == "vertical":
            for i in range(height):
                ratio = i / height
                r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
                g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
                b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
                a = int(start_color[3] * (1 - ratio) + end_color[3] * ratio)
                draw.line([(0, i), (width, i)], fill=(r, g, b, a))
        elif direction == "horizontal":
            for i in range(width):
                ratio = i / width
                r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
                g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
                b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
                a = int(start_color[3] * (1 - ratio) + end_color[3] * ratio)
                draw.line([(i, 0), (i, height)], fill=(r, g, b, a))
        
        self.current_layer.paste(gradient, (x, y), gradient)
    
    def draw_radial_gradient(self, x: int, y: int, radius: int,
                           center_color: Tuple[int, int, int, int],
                           edge_color: Tuple[int, int, int, int]):
        """绘制径向渐变"""
        size = radius * 2
        gradient = Image.new("RGBA", (size, size))
        draw = ImageDraw.Draw(gradient)
        
        for r in range(radius, 0, -1):
            ratio = r / radius
            rc = int(center_color[0] * ratio + edge_color[0] * (1 - ratio))
            gc = int(center_color[1] * ratio + edge_color[1] * (1 - ratio))
            bc = int(center_color[2] * ratio + edge_color[2] * (1 - ratio))
            ac = int(center_color[3] * ratio + edge_color[3] * (1 - ratio))
            draw.ellipse([radius - r, radius - r, radius + r, radius + r], 
                        fill=(rc, gc, bc, ac))
        
        self.current_layer.paste(gradient, (x - radius, y - radius), gradient)


# 示例测试
if __name__ == "__main__":
    renderer = AdvancedRendererV2(800, 600, bg_color=(10, 20, 40))
    
    # 添加光源
    renderer.add_light_source(600, 100, intensity=0.8, color=(255, 255, 200))
    
    # 背景渐变
    renderer.draw_linear_gradient(0, 0, 800, 600,
                                 start_color=(10, 20, 40, 255),
                                 end_color=(30, 50, 80, 255))
    
    # 3D球体
    renderer.create_layer()
    renderer.draw_3d_sphere(200, 300, 80, color=(255, 100, 100, 255))
    
    # 3D立方体
    renderer.create_layer()
    renderer.draw_3d_cube(400, 300, 60, rotation=0.3)
    
    # 粒子系统
    renderer.create_layer()
    renderer.draw_particle_system(600, 300, count=80, particle_color=(255, 200, 100, 200), spread=80)
    
    # 应用效果
    renderer.apply_bloom(threshold=180, blur_radius=8.0)
    renderer.apply_vignette(intensity=0.4)
    
    # 保存
    final = renderer.merge_layers()
    final.save("渲染引擎v2测试.png")
    print("✅ 高级渲染引擎v2测试完成，图片已保存为 渲染引擎v2测试.png")
