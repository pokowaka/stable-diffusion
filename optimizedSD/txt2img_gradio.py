import argparse
import asyncio
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import gradio as gr
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging as transformers_logging
import mimetypes
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts

transformers_logging.set_verbosity_error()

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


async def get_logs():
    return "\n".join([x for x in open("log.txt", "r", encoding="utf8").readlines()] +
                     [y for y in open("tqdm.txt", "r", encoding="utf8").readlines()])


async def get_nvidia_smi():
    proc = await asyncio.create_subprocess_shell('nvidia-smi', stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return str(stdout)

def generate(
        prompt,
        ddim_steps,
        n_iter,
        batch_size,
        Height,
        Width,
        scale,
        ddim_eta,
        unet_bs,
        device,
        seed,
        outdir,
        img_format,
        turbo,
        full_precision,
        sampler,
        speed_mp
):
    logging.info(f"prompt: {prompt}, W: {Width}, H: {Height}")
    C = 4
    f = 8
    start_code = None
    model.to(device)
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)

    if device != "cpu" and not full_precision:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt.replace("/", ""))))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if device != "cpu" and not full_precision:
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    with torch.no_grad():
        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler=sampler,
                        speed_mp=speed_mp
                    )

                    modelFS.to(device)
                    model.cpu()
                    logging.info("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                        )
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    logging.info(str("memory_final = " + str(torch.cuda.memory_allocated() / 1e6)))

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to "
            + sample_path
            + "\nSeeds used = "
            + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), txt


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


if __name__ == '__main__':
    global lines
    lines = []
    file_handler = logging.FileHandler(filename='log.txt', mode='w')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler, TqdmLoggingHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    parser = argparse.ArgumentParser(description='txt2img using gradio')
    parser.add_argument('--config_path', default="optimizedSD/v1-inference.yaml", type=str, help='config path')
    parser.add_argument('--ckpt_path', default="models/ldm/stable-diffusion-v1/model.ckpt", type=str, help='ckpt path')
    parser.add_argument('--outputs_path', default="outputs/txt2img-samples", type=str, help='output imgs path')
    args = parser.parse_args()
    config = args.config_path
    ckpt = args.ckpt_path
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    demo = gr.Blocks()

    with demo:
        with gr.Column():
            gr.Markdown("# Stable diffusion txt2img (neonsecret's adjustments)")
            gr.Markdown("### Press 'print logs' button to get the model output logs")
            with gr.Row():
                with gr.Column():
                    outs1 = [gr.Image(label="Output Image"), gr.Text(label="Generation results")]
                    outs2 = [gr.Text(label="Logs")]
                    outs3 = [gr.Text(label="nvidia-smi")]
                    b1 = gr.Button("Generate!")
                    b2 = gr.Button("Print logs")
                    b3 = gr.Button("nvidia-smi")
                with gr.Column():
                    with gr.Box():
                        b1.click(generate, inputs=[
                            gr.Text(label="Your Prompt"),
                            gr.Slider(1, 1000, value=50, label="Sampling Steps"),
                            gr.Slider(1, 100, step=1, label="Number of images"),
                            gr.Slider(1, 100, step=1, label="Batch size"),
                            gr.Slider(64, 4096, value=512, step=64, label="Height"),
                            gr.Slider(64, 4096, value=512, step=64, label="Width"),
                            gr.Slider(0, 50, value=7.5, step=0.1, label="Guidance scale"),
                            gr.Slider(0, 1, step=0.01, label="DDIM sampling ETA"),
                            gr.Slider(1, 2, value=1, step=1, label="U-Net batch size"),
                            gr.Radio(["cuda", "cpu"], value="cuda", label="Device"),
                            gr.Text(label="Seed"),
                            gr.Text(value=args.outputs_path, label="Outputs path"),
                            gr.Radio(["png", "jpg"], value='png', label="Image format"),
                            gr.Checkbox(value=True, label="Turbo mode (better leave this on)"),
                            gr.Checkbox(label="Full precision mode (practically does nothing)"),
                            gr.Radio(["ddim", "plms", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"], value="plms", label="Sampler"),
                            gr.Slider(1, 10, value=1, step=1, label="speed_mp multiplier (don't change if not sure)"),
                        ], outputs=outs1)
                        b2.click(get_logs, inputs=[], outputs=outs2)
                        b3.click(get_nvidia_smi, inputs=[], outputs=outs3)

    demo.launch(share=True)
