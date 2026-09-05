from __future__ import annotations

import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


class ComfyUIClient:
    def __init__(self, api_url: str = "", output_dir: str = "outputs") -> None:
        self.api_url = str(api_url or "").strip().rstrip("/")
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._is_comfyui: Optional[bool] = None
        self._checkpoint_cache: list[str] | None = None
        self._lora_cache: list[str] | None = None
        self._controlnet_cache: list[str] | None = None

    def set_api_url(self, api_url: str) -> None:
        self.api_url = str(api_url or "").strip().rstrip("/")
        self._is_comfyui = None
        self._checkpoint_cache = None
        self._lora_cache = None
        self._controlnet_cache = None

    @property
    def available(self) -> bool:
        return self.api_url.startswith("http")

    def is_comfyui_server(self) -> bool:
        if not self.available:
            return False
        if self._is_comfyui is not None:
            return self._is_comfyui
        try:
            response = requests.get(f"{self.api_url}/system_stats", timeout=5)
            self._is_comfyui = response.ok and "application/json" in response.headers.get("Content-Type", "")
        except Exception:
            self._is_comfyui = False
        return bool(self._is_comfyui)

    def _sampler_config(self, sampler_name: str | None = None) -> tuple[str, str]:
        sampler = str(sampler_name or "").strip().lower()
        mapping = {
            "dpm++ 2m karras": ("dpmpp_2m", "karras"),
            "dpm++ sde karras": ("dpmpp_sde", "karras"),
            "euler": ("euler", "normal"),
            "euler a": ("euler_ancestral", "normal"),
            "heun": ("heun", "normal"),
            "ddim": ("ddim", "normal"),
        }
        return mapping.get(sampler, ("dpmpp_2m", "karras"))

    def _sanitize_dimension(self, value: int, minimum: int = 512) -> int:
        value = max(minimum, int(value or minimum))
        return max(64, (value // 64) * 64)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        response = requests.request(method, f"{self.api_url}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _extract_choice_names(self, payload: Dict[str, Any], node_name: str, field_name: str) -> list[str]:
        if node_name in payload and isinstance(payload[node_name], dict):
            payload = payload[node_name]
        required = ((payload.get("input") or {}).get("required") or {}).get(field_name)
        if isinstance(required, list) and required and isinstance(required[0], list):
            return [str(item) for item in required[0] if str(item).strip() and "put_" not in str(item)]
        if isinstance(required, list):
            return [str(item) for item in required if str(item).strip() and "put_" not in str(item)]
        return []

    def get_available_checkpoints(self) -> list[str]:
        if self._checkpoint_cache is not None:
            return list(self._checkpoint_cache)
        if not self.is_comfyui_server():
            self._checkpoint_cache = []
            return []
        try:
            payload = self._request_json("GET", "/object_info/CheckpointLoaderSimple", timeout=15)
        except Exception:
            self._checkpoint_cache = []
            return []
        self._checkpoint_cache = self._extract_choice_names(payload, "CheckpointLoaderSimple", "ckpt_name")
        return list(self._checkpoint_cache)

    def get_available_loras(self) -> list[str]:
        if self._lora_cache is not None:
            return list(self._lora_cache)
        if not self.is_comfyui_server():
            self._lora_cache = []
            return []
        try:
            payload = self._request_json("GET", "/object_info/LoraLoader", timeout=15)
        except Exception:
            self._lora_cache = []
            return []
        self._lora_cache = self._extract_choice_names(payload, "LoraLoader", "lora_name")
        return list(self._lora_cache)

    def get_available_controlnets(self) -> list[str]:
        if self._controlnet_cache is not None:
            return list(self._controlnet_cache)
        if not self.is_comfyui_server():
            self._controlnet_cache = []
            return []
        try:
            payload = self._request_json("GET", "/object_info/ControlNetLoader", timeout=15)
        except Exception:
            self._controlnet_cache = []
            return []
        self._controlnet_cache = self._extract_choice_names(payload, "ControlNetLoader", "control_net_name")
        return list(self._controlnet_cache)

    def pick_checkpoint(self, preferred_name: str | None = None) -> str:
        checkpoints = self.get_available_checkpoints()
        if preferred_name:
            marker = str(preferred_name).strip().casefold()
            for name in checkpoints:
                if name.casefold() == marker:
                    return name
            for name in checkpoints:
                if marker in name.casefold():
                    return name
        for matcher in ("sd_xl_base_1.0", "xl_base", "sdxl", "base"):
            for name in checkpoints:
                if matcher in name.casefold():
                    return name
        if checkpoints:
            return checkpoints[0]
        raise RuntimeError("ComfyUI 未返回可用 checkpoint")

    def pick_lora(self, preferred_name: str | None = None) -> str:
        loras = self.get_available_loras()
        if preferred_name:
            marker = str(preferred_name).strip().casefold()
            for name in loras:
                if name.casefold() == marker:
                    return name
            for name in loras:
                if marker in name.casefold():
                    return name
        if loras:
            return loras[0]
        raise RuntimeError("ComfyUI 未返回可用 LoRA")

    def pick_controlnet(self, preferred_name: str | list[str] | None = None) -> str:
        controlnets = self.get_available_controlnets()
        preferred: list[str] = []
        if isinstance(preferred_name, list):
            preferred = [str(item).strip().casefold() for item in preferred_name if str(item).strip()]
        elif preferred_name:
            preferred = [str(preferred_name).strip().casefold()]
        for marker in preferred:
            for name in controlnets:
                if name.casefold() == marker:
                    return name
            for name in controlnets:
                if marker in name.casefold():
                    return name
        for matcher in ("canny", "sketch", "scribble", "lineart", "depth"):
            for name in controlnets:
                if matcher in name.casefold():
                    return name
        if controlnets:
            return controlnets[0]
        raise RuntimeError("ComfyUI 未返回可用 ControlNet/T2I 模型")

    def upload_image(self, image_path: str, *, overwrite: bool = True) -> str:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as handle:
            response = requests.post(
                f"{self.api_url}/upload/image",
                files={"image": (os.path.basename(image_path), handle, mime_type)},
                data={"overwrite": "true" if overwrite else "false"},
                timeout=180,
            )
        response.raise_for_status()
        payload = response.json()
        name = str(payload.get("name") or "").strip()
        subfolder = str(payload.get("subfolder") or "").strip()
        return f"{subfolder}/{name}".strip("/") if subfolder else name

    def _submit_prompt(self, workflow: Dict[str, Any]) -> str:
        payload = {
            "prompt": workflow,
            "client_id": uuid.uuid4().hex,
        }
        response = requests.post(f"{self.api_url}/prompt", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        prompt_id = str(result.get("prompt_id") or "").strip()
        if not prompt_id:
            raise RuntimeError("ComfyUI 未返回 prompt_id")
        return prompt_id

    def _history_outputs(self, prompt_id: str) -> Dict[str, Any]:
        payload = self._request_json("GET", f"/history/{quote(prompt_id, safe='')}", timeout=30)
        if prompt_id in payload and isinstance(payload[prompt_id], dict):
            return payload[prompt_id]
        return payload

    def _wait_for_image_descriptor(self, prompt_id: str, *, timeout_seconds: int = 600) -> Dict[str, Any]:
        deadline = time.time() + max(30, timeout_seconds)
        while time.time() < deadline:
            history = self._history_outputs(prompt_id)
            outputs = history.get("outputs") or {}
            for node_output in outputs.values():
                if not isinstance(node_output, dict):
                    continue
                images = node_output.get("images") or []
                if images and isinstance(images[0], dict):
                    return images[0]
            status = history.get("status") or {}
            completed = ((status.get("status_str") or "") == "success") or bool(status.get("completed") is True)
            if completed:
                break
            time.sleep(1.2)
        raise TimeoutError("ComfyUI 生成超时或未返回图片")

    def _download_image(self, descriptor: Dict[str, Any], output_name: str) -> str:
        filename = str(descriptor.get("filename") or "").strip()
        if not filename:
            raise RuntimeError("ComfyUI 返回图片描述缺少 filename")
        params = {
            "filename": filename,
            "subfolder": str(descriptor.get("subfolder") or ""),
            "type": str(descriptor.get("type") or "output"),
        }
        response = requests.get(f"{self.api_url}/view", params=params, timeout=180)
        response.raise_for_status()
        extension = os.path.splitext(filename)[1] or ".png"
        output_path = os.path.join(self.output_dir, f"{output_name}{extension}")
        with open(output_path, "wb") as handle:
            handle.write(response.content)
        return output_path

    def _workflow_refs(
        self,
        workflow: Dict[str, Any],
        *,
        checkpoint: str,
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        lora_node_id: str = "90",
    ) -> tuple[list[Any], list[Any]]:
        workflow["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}}
        if str(lora_name or "").strip():
            workflow[lora_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": self.pick_lora(lora_name),
                    "strength_model": float(lora_strength_model),
                    "strength_clip": float(lora_strength_clip),
                },
            }
            return [lora_node_id, 0], [lora_node_id, 1]
        return ["1", 0], ["1", 1]

    def render_img2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image_path: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        denoising_strength: float,
        sampler_name: str = "",
        filename_prefix: str = "comfy_img2img",
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        uploaded_name = self.upload_image(control_image_path)
        checkpoint = self.pick_checkpoint(checkpoint_name)
        sampler, scheduler = self._sampler_config(sampler_name)
        workflow: Dict[str, Any] = {}
        model_ref, clip_ref = self._workflow_refs(
            workflow,
            checkpoint=checkpoint,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )
        workflow.update(
            {
                "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_name}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_ref}},
                "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
                "6": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 2**31 - 1),
                        "steps": max(1, int(steps)),
                        "cfg": float(cfg_scale),
                        "sampler_name": sampler,
                        "scheduler": scheduler,
                        "denoise": max(0.0, min(1.0, float(denoising_strength))),
                        "model": model_ref,
                        "positive": ["3", 0],
                        "negative": ["4", 0],
                        "latent_image": ["5", 0],
                    },
                },
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["7", 0]}},
            }
        )
        prompt_id = self._submit_prompt(workflow)
        descriptor = self._wait_for_image_descriptor(prompt_id)
        return self._download_image(descriptor, f"{filename_prefix}_{prompt_id}")

    def render_controlnet_img2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        init_image_path: str,
        control_inputs: list[Dict[str, Any]],
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        denoising_strength: float,
        sampler_name: str = "",
        filename_prefix: str = "comfy_controlnet_img2img",
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        normalized_inputs = [dict(item) for item in control_inputs if isinstance(item, dict) and str(item.get("image_path") or "").strip()]
        if not normalized_inputs:
            return self.render_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image_path=init_image_path,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                denoising_strength=denoising_strength,
                sampler_name=sampler_name,
                filename_prefix=filename_prefix,
                checkpoint_name=checkpoint_name,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
            )

        uploaded_init = self.upload_image(init_image_path)
        checkpoint = self.pick_checkpoint(checkpoint_name)
        sampler, scheduler = self._sampler_config(sampler_name)
        workflow: Dict[str, Any] = {}
        model_ref, clip_ref = self._workflow_refs(
            workflow,
            checkpoint=checkpoint,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )
        workflow.update(
            {
                "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_init}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_ref}},
                "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
            }
        )

        positive_ref: list[Any] = ["3", 0]
        negative_ref: list[Any] = ["4", 0]
        next_node_id = 20
        for item in normalized_inputs:
            uploaded_control = self.upload_image(str(item.get("image_path") or "").strip())
            control_model = self.pick_controlnet(item.get("control_net_name") or item.get("controlnet_name"))
            load_image_id = str(next_node_id)
            load_controlnet_id = str(next_node_id + 1)
            apply_id = str(next_node_id + 2)
            next_node_id += 3
            workflow[load_image_id] = {"class_type": "LoadImage", "inputs": {"image": uploaded_control}}
            workflow[load_controlnet_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": control_model}}
            workflow[apply_id] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": positive_ref,
                    "negative": negative_ref,
                    "control_net": [load_controlnet_id, 0],
                    "image": [load_image_id, 0],
                    "strength": float(item.get("strength", 0.8)),
                    "start_percent": float(item.get("start_percent", 0.0)),
                    "end_percent": float(item.get("end_percent", 1.0)),
                    "vae": ["1", 2],
                },
            }
            positive_ref = [apply_id, 0]
            negative_ref = [apply_id, 1]

        workflow["6"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": random.randint(1, 2**31 - 1),
                "steps": max(1, int(steps)),
                "cfg": float(cfg_scale),
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": max(0.0, min(1.0, float(denoising_strength))),
                "model": model_ref,
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": ["5", 0],
            },
        }
        workflow["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
        workflow["8"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["7", 0]}}
        prompt_id = self._submit_prompt(workflow)
        descriptor = self._wait_for_image_descriptor(prompt_id)
        return self._download_image(descriptor, f"{filename_prefix}_{prompt_id}")

    def render_controlnet_txt2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_inputs: list[Dict[str, Any]],
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler_name: str = "",
        filename_prefix: str = "comfy_controlnet_txt2img",
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        normalized_inputs = [dict(item) for item in control_inputs if isinstance(item, dict) and str(item.get("image_path") or "").strip()]
        if not normalized_inputs:
            return self.render_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                filename_prefix=filename_prefix,
                checkpoint_name=checkpoint_name,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
            )

        checkpoint = self.pick_checkpoint(checkpoint_name)
        sampler, scheduler = self._sampler_config(sampler_name)
        workflow: Dict[str, Any] = {}
        model_ref, clip_ref = self._workflow_refs(
            workflow,
            checkpoint=checkpoint,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )
        workflow.update(
            {
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_ref}},
            }
        )

        positive_ref: list[Any] = ["2", 0]
        negative_ref: list[Any] = ["3", 0]
        next_node_id = 20
        for item in normalized_inputs:
            uploaded_control = self.upload_image(str(item.get("image_path") or "").strip())
            control_model = self.pick_controlnet(item.get("control_net_name") or item.get("controlnet_name"))
            load_image_id = str(next_node_id)
            load_controlnet_id = str(next_node_id + 1)
            apply_id = str(next_node_id + 2)
            next_node_id += 3
            workflow[load_image_id] = {"class_type": "LoadImage", "inputs": {"image": uploaded_control}}
            workflow[load_controlnet_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": control_model}}
            workflow[apply_id] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": positive_ref,
                    "negative": negative_ref,
                    "control_net": [load_controlnet_id, 0],
                    "image": [load_image_id, 0],
                    "strength": float(item.get("strength", 0.8)),
                    "start_percent": float(item.get("start_percent", 0.0)),
                    "end_percent": float(item.get("end_percent", 1.0)),
                    "vae": ["1", 2],
                },
            }
            positive_ref = [apply_id, 0]
            negative_ref = [apply_id, 1]

        workflow["4"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": self._sanitize_dimension(width),
                "height": self._sanitize_dimension(height),
                "batch_size": 1,
            },
        }
        workflow["5"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": random.randint(1, 2**31 - 1),
                "steps": max(1, int(steps)),
                "cfg": float(cfg_scale),
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": model_ref,
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": ["4", 0],
            },
        }
        workflow["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}}
        workflow["7"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["6", 0]}}
        prompt_id = self._submit_prompt(workflow)
        descriptor = self._wait_for_image_descriptor(prompt_id)
        return self._download_image(descriptor, f"{filename_prefix}_{prompt_id}")

    def render_inpaint(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        init_image_path: str,
        mask_image_path: str,
        steps: int,
        cfg_scale: float,
        denoising_strength: float,
        sampler_name: str = "",
        filename_prefix: str = "comfy_inpaint",
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
        grow_mask_by: int = 12,
    ) -> str:
        uploaded_init = self.upload_image(init_image_path)
        uploaded_mask = self.upload_image(mask_image_path)
        checkpoint = self.pick_checkpoint(checkpoint_name)
        sampler, scheduler = self._sampler_config(sampler_name)
        workflow: Dict[str, Any] = {}
        model_ref, clip_ref = self._workflow_refs(
            workflow,
            checkpoint=checkpoint,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )
        workflow.update(
            {
                "2": {"class_type": "LoadImage", "inputs": {"image": uploaded_init}},
                "3": {"class_type": "LoadImage", "inputs": {"image": uploaded_mask}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_ref}},
                "6": {
                    "class_type": "VAEEncodeForInpaint",
                    "inputs": {
                        "pixels": ["2", 0],
                        "mask": ["3", 1],
                        "vae": ["1", 2],
                        "grow_mask_by": max(0, int(grow_mask_by)),
                    },
                },
                "7": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 2**31 - 1),
                        "steps": max(1, int(steps)),
                        "cfg": float(cfg_scale),
                        "sampler_name": sampler,
                        "scheduler": scheduler,
                        "denoise": max(0.0, min(1.0, float(denoising_strength))),
                        "model": model_ref,
                        "positive": ["4", 0],
                        "negative": ["5", 0],
                        "latent_image": ["6", 0],
                    },
                },
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
            }
        )
        prompt_id = self._submit_prompt(workflow)
        descriptor = self._wait_for_image_descriptor(prompt_id)
        return self._download_image(descriptor, f"{filename_prefix}_{prompt_id}")

    def render_txt2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler_name: str = "",
        filename_prefix: str = "comfy_txt2img",
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        checkpoint = self.pick_checkpoint(checkpoint_name)
        sampler, scheduler = self._sampler_config(sampler_name)
        workflow: Dict[str, Any] = {}
        model_ref, clip_ref = self._workflow_refs(
            workflow,
            checkpoint=checkpoint,
            lora_name=lora_name,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
        )
        workflow.update(
            {
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_ref}},
                "4": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "width": self._sanitize_dimension(width),
                        "height": self._sanitize_dimension(height),
                        "batch_size": 1,
                    },
                },
                "5": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 2**31 - 1),
                        "steps": max(1, int(steps)),
                        "cfg": float(cfg_scale),
                        "sampler_name": sampler,
                        "scheduler": scheduler,
                        "denoise": 1.0,
                        "model": model_ref,
                        "positive": ["2", 0],
                        "negative": ["3", 0],
                        "latent_image": ["4", 0],
                    },
                },
                "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
                "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["6", 0]}},
            }
        )
        prompt_id = self._submit_prompt(workflow)
        descriptor = self._wait_for_image_descriptor(prompt_id)
        return self._download_image(descriptor, f"{filename_prefix}_{prompt_id}")
