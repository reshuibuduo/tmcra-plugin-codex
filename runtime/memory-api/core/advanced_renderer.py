"""
Tri-Maze 高级渲染引擎 v1.0
完善基础渲染能力，支持渐变、阴影、模糊、纹理、图层等效果
完全模块化，与核心推理逻辑解耦
"""
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import numpy as np
import math
from typing import Tuple, List, Dict

class AdvancedRenderer:
    """
    高级渲染引擎，支持丰富的图形效果
    完全独立于核心推理逻辑，可单独升级完善
    """
    
    def __init__(self, width: int = 1024, height: int = 768, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.layers = []
        self.current_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.current_layer)
        self.bg_color = bg_color + (255,)
    
    def create_layer(self) -> int:
        """创建新图层"""
        self.layers.append(self.current_layer)
        self.current_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.current_layer)
        return len(self.layers)
    
    def merge_layers(self) -> Image.Image:
        """合并所有图层"""
        final = Image.new("RGBA", (self.width, self.height), self.bg_color)
        for layer in self.layers + [self.current_layer]:
            final = Image.alpha_composite(final, layer)
        return final.convert("RGB")
    
    # ===== 基础形状绘制 =====
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
    
    def draw_ellipse(self, x: int, y: int, width: int, height: int,
                    fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
                    outline: Tuple[int, int, int, int] = None,
                    stroke_width: int = 1):
        """绘制椭圆"""
        self.draw.ellipse([x, y, x + width, y + height], 
                         fill=fill, outline=outline, width=stroke_width)
    
    def draw_polygon(self, points: List[Tuple[int, int]],
                    fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
                    outline: Tuple[int, int, int, int] = None,
                    stroke_width: int = 1):
        """绘制多边形"""
        self.draw.polygon(points, fill=fill, outline=outline, width=stroke_width)
    
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int],
                 color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                 width: int = 1,
                 dashed: bool = False,
                 dash_length: int = 10):
        """绘制直线，支持虚线"""
        if not dashed:
            self.draw.line([start, end], fill=color, width=width)
            return
        
        # 虚线
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        dashes = int(distance / dash_length)
        
        for i in range(dashes):
            if i % 2 == 0:
                s = i / dashes
                e = (i + 1) / dashes
                sx = x1 + dx * s
                sy = y1 + dy * s
                ex = x1 + dx * e
                ey = y1 + dy * e
                self.draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
    
    def draw_bezier_curve(self, points: List[Tuple[int, int]],
                         color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                         width: int = 1,
                         segments: int = 100):
        """绘制贝塞尔曲线"""
        if len(points) < 2:
            return
        
        def bezier(t, points):
            n = len(points) - 1
            x = 0
            y = 0
            for i, (px, py) in enumerate(points):
                binom = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
                x += px * binom
                y += py * binom
            return (int(x), int(y))
        
        curve_points = []
        for t in np.linspace(0, 1, segments):
            curve_points.append(bezier(t, points))
        
        for i in range(len(curve_points) - 1):
            self.draw.line([curve_points[i], curve_points[i+1]], fill=color, width=width)
    
    # ===== 渐变效果 =====
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
        """绘制径向渐变（光晕效果）"""
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
    
    # ===== 效果滤镜 =====
    def apply_blur(self, radius: float = 2.0, layer_index: int = None):
        """应用模糊效果"""
        if layer_index is None:
            self.current_layer = self.current_layer.filter(ImageFilter.GaussianBlur(radius))
        else:
            self.layers[layer_index] = self.layers[layer_index].filter(ImageFilter.GaussianBlur(radius))
    
    def apply_shadow(self, offset: Tuple[int, int] = (5, 5), blur_radius: float = 5.0, 
                    color: Tuple[int, int, int, int] = (0, 0, 0, 100),
                    layer_index: int = None):
        """应用阴影效果"""
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        
        # 创建阴影层
        shadow = Image.new("RGBA", target_layer.size, (0, 0, 0, 0))
        alpha = target_layer.getchannel("A")
        shadow.paste(color, mask=alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # 合并阴影和原图层
        result = Image.new("RGBA", target_layer.size, (0, 0, 0, 0))
        result.paste(shadow, offset, shadow)
        result = Image.alpha_composite(result, target_layer)
        
        if layer_index is None:
            self.current_layer = result
        else:
            self.layers[layer_index] = result
    
    def apply_glow(self, blur_radius: float = 10.0, color: Tuple[int, int, int, int] = (255, 255, 200, 150),
                  layer_index: int = None):
        """应用发光效果"""
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        
        # 创建发光层
        glow = Image.new("RGBA", target_layer.size, (0, 0, 0, 0))
        alpha = target_layer.getchannel("A")
        glow.paste(color, mask=alpha)
        glow = glow.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # 合并发光和原图层
        result = Image.alpha_composite(glow, target_layer)
        
        if layer_index is None:
            self.current_layer = result
        else:
            self.layers[layer_index] = result
    
    def apply_noise(self, amount: float = 0.1, monochrome: bool = False, layer_index: int = None):
        """应用噪点纹理效果"""
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        np_img = np.array(target_layer)
        
        if monochrome:
            noise = np.random.normal(0, amount * 255, np_img.shape[:2])
            np_img[..., :3] = np.clip(np_img[..., :3] + noise[..., np.newaxis], 0, 255)
        else:
            noise = np.random.normal(0, amount * 255, np_img.shape)
            np_img[..., :3] = np.clip(np_img[..., :3] + noise[..., :3], 0, 255)
        
        result = Image.fromarray(np_img.astype(np.uint8), "RGBA")
        
        if layer_index is None:
            self.current_layer = result
        else:
            self.layers[layer_index] = result
    
    # ===== 变换 =====
    def rotate_layer(self, angle: float, expand: bool = False, layer_index: int = None):
        """旋转图层"""
        if layer_index is None:
            self.current_layer = self.current_layer.rotate(angle, expand=expand, resample=Image.Resampling.BILINEAR)
        else:
            self.layers[layer_index] = self.layers[layer_index].rotate(angle, expand=expand, resample=Image.Resampling.BILINEAR)
    
    def scale_layer(self, scale_x: float, scale_y: float = None, layer_index: int = None):
        """缩放图层"""
        if scale_y is None:
            scale_y = scale_x
        
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        new_width = int(target_layer.width * scale_x)
        new_height = int(target_layer.height * scale_y)
        resized = target_layer.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        
        if layer_index is None:
            self.current_layer = resized
        else:
            self.layers[layer_index] = resized
    
    def translate_layer(self, dx: int, dy: int, layer_index: int = None):
        """平移图层"""
        target_layer = self.current_layer if layer_index is None else self.layers[layer_index]
        translated = Image.new("RGBA", target_layer.size, (0, 0, 0, 0))
        translated.paste(target_layer, (dx, dy), target_layer)
        
        if layer_index is None:
            self.current_layer = translated
        else:
            self.layers[layer_index] = translated


# 示例使用
if __name__ == "__main__":
    # 创建渲染器
    renderer = AdvancedRenderer(800, 600, bg_color=(10, 20, 40))
    
    # 背景渐变
    renderer.draw_linear_gradient(0, 0, 800, 600,
                                 start_color=(10, 20, 40, 255),
                                 end_color=(30, 50, 80, 255),
                                 direction="vertical")
    
    # 新建图层画月亮
    renderer.create_layer()
    renderer.draw_radial_gradient(650, 150, 80,
                                 center_color=(255, 255, 220, 255),
                                 edge_color=(255, 255, 220, 0))
    renderer.draw_circle(650, 150, 50, fill=(255, 255, 220, 255))
    
    # 新建图层画荷花
    renderer.create_layer()
    # 花茎
    renderer.draw_line((200, 300), (200, 500), color=(0, 100, 0, 255), width=4)
    # 花瓣
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x = 200 + math.cos(rad) * 40
        y = 300 + math.sin(rad) * 60
        renderer.draw_ellipse(x - 15, y - 30, x + 15, y + 30,
                             fill=(255, 150, 180, 220),
                             outline=(200, 80, 120, 255),
                             stroke_width=1)
    # 花心
    renderer.draw_circle(200, 300, 20, fill=(255, 200, 100, 255))
    
    # 应用阴影
    renderer.apply_shadow(offset=(3, 3), blur_radius=5.0, layer_index=1)
    
    # 合并图层并保存
    final_image = renderer.merge_layers()
    final_image.save("渲染引擎测试.png")
    print("✅ 高级渲染引擎测试完成，图片已保存为 渲染引擎测试.png")
