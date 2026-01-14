import torch
from tqdm import tqdm
from PIL import Image


def generate_lcm_tfg(worker, prompt, num_inference_steps, tfg_scale, n_recur, loss_fn, seed):
        prompt_embeds = worker.get_prompt_embeds(prompt)
        worker.scheduler.set_timesteps(num_inference_steps)
        timesteps = worker.scheduler.timesteps.to(worker.device_0)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=worker.device_0).manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        latents = torch.randn(
            (1, worker.unet.config.in_channels, 64, 64),
            device=worker.device_0,
            dtype=torch.float32,
            generator=generator
        )
        latents = latents * worker.scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):
            if loss_fn is not None:
                cur_step_idx = worker.scheduler._step_index 
                for r in range(n_recur):
                    with torch.enable_grad():
                        latents = latents.detach().requires_grad_(True)
                        model_pred = worker.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
                        c_skip, c_out = worker.scheduler.get_scalings_for_boundary_condition_discrete(t)
                        pred_x0 = c_skip * latents + c_out * model_pred
                        
                        pred_x0_pixel = worker.vae.decode(pred_x0.to(worker.device_1) / worker.vae.config.scaling_factor).sample
                        loss = loss_fn(pred_x0_pixel)
                        grad = torch.autograd.grad(loss, latents)[0]

                        with torch.no_grad():
                            worker.scheduler._step_index = cur_step_idx
                            latents_next = worker.scheduler.step(model_pred, t, latents, return_dict=True).prev_sample

                        alpha_prod_t = worker.scheduler.alphas_cumprod[t.long()]
                        scale_factor = tfg_scale / (alpha_prod_t ** 0.5)
                        latents_next = latents_next + scale_factor * grad
                    
                    if r < n_recur - 1:
                        with torch.no_grad():
                            if i + 1 < len(timesteps):
                                next_t = timesteps[i + 1]
                                alpha_prod_t_next = worker.scheduler.alphas_cumprod[next_t.long()]
                                ratio = alpha_prod_t / alpha_prod_t_next
                                
                                noise = torch.randn(
                                    latents.shape, 
                                    device=latents.device, 
                                    dtype=latents.dtype, 
                                    generator=generator 
                                )
                                
                                latents = (ratio ** 0.5) * latents_next + ((1 - ratio) ** 0.5) * noise
                            else:
                                latents = latents_next
                    else:
                        latents = latents_next
                
            else:
                with torch.no_grad():
                    model_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
                    latents = scheduler.step(model_pred, t, latents, return_dict=True).prev_sample
            
        with torch.no_grad():
            image = worker.vae.decode(latents.to(worker.device_1) / worker.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype("uint8")
            return Image.fromarray(image[0])