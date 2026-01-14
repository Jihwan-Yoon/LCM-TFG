import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from src.pipeline import generate_lcm_tfg

class LCMWorker:
    def __init__(self, device_unet, device_vae, model_id="SimianLuo/LCM_Dreamshaper_v7"):
        self.device_0 = device_unet
        self.device_1 = device_vae
        
        self.scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device_0, dtype=torch.float32)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device_1, dtype=torch.float32)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device_0, dtype=torch.float32)
        self.unet.enable_gradient_checkpointing()

    def get_prompt_embeds(self, prompt):
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device_0))[0]
        return prompt_embeds

    def generate(self, **kwargs):
        return generate_lcm_tfg(self, **kwargs)
    