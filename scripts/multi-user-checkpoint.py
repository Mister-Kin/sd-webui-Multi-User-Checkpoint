import gradio as gr
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import modules.shared as shared
import shutil

from modules import scripts
from modules.ui import refresh_symbol
from modules.ui_components import ToolButton
from modules.sd_models import select_checkpoint, load_model
from modules.shared_items import sd_vae_items, refresh_vae_list
from modules.shared import opts, list_checkpoint_tiles, refresh_checkpoints
from modules.sd_models import get_closet_checkpoint_match


class MultiUserCKPT(scripts.Script):
    def title(self):
        return "Multi User Checkpoint"

    def describe(self):
        return (
            "Allow multiple clients to create task queues with different checkpoints."
        )

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return [self.checkpoint, self.vae]

    def refresh_ckpts(self):
        refresh_checkpoints()
        choices = list_checkpoint_tiles()
        return {"choices": choices, "__type__": "update"}

    def refresh_vaes(self):
        refresh_vae_list()
        choices = sd_vae_items()
        return {"choices": choices, "__type__": "update"}

    def before_component(self, component, **kwargs):
        if kwargs.get("label") == f"Sampling method":
            ckpt_choices = list_checkpoint_tiles()
            self.checkpoint = gr.Dropdown(
                elem_id="muc_checkpoint",
                label="你的「Stable Diffusion 模型」",
                value="不更改",
                choices=["不更改"] + ckpt_choices,
                interactive=True,
            )
            self.refresh_ckpt = ToolButton(
                value=refresh_symbol, elem_id="muc_refresh_ckpt"
            )
            self.refresh_ckpt.click(
                fn=self.refresh_ckpts, inputs=[], outputs=[self.checkpoint]
            )

            vae_choices = sd_vae_items()
            self.vae = gr.Dropdown(
                elem_id="muc_vae",
                label="你的「外挂 VAE 模型」",
                value="不更改",
                choices=["不更改"] + vae_choices,
                interactive=True,
            )
            self.refresh_vae = ToolButton(
                value=refresh_symbol, elem_id="muc_refresh_vae"
            )
            self.refresh_vae.click(fn=self.refresh_vaes, inputs=[], outputs=[self.vae])

    def process(self, p, ckpt: str, vae: str):
        # if vae != opts.sd_vae and vae != "Do not change":
        #     opts.sd_vae = vae
        # if ckpt != opts.sd_model_checkpoint and ckpt != "Do not change":
        #     opts.sd_model_checkpoint = ckpt
        #     load_model(checkpoint_info=select_checkpoint())
        terminal_width = shutil.get_terminal_size().columns

        if vae != opts.sd_vae and vae != "不更改":
            print("\n")
            print("=" * terminal_width)
            print("「外挂 VAE 模型」当前为：", opts.sd_vae)
            print("-" * terminal_width)
            opts.sd_vae = vae
            p.sd_vae_name = vae
            p.override_settings["sd_vae"] = vae
            sd_vae.reload_vae_weights()
            print("-" * terminal_width)
            print("「外挂 VAE 模型」已切换为：", opts.sd_vae)
            print("=" * terminal_width)
            print("\n")
        if ckpt != opts.sd_model_checkpoint and ckpt != "不更改":
            print("\n")
            print("=" * terminal_width)
            print("「Stable Diffusion 模型」当前为：", opts.sd_model_checkpoint)
            print("-" * terminal_width)
            info = get_closet_checkpoint_match(ckpt)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {ckpt}")
            p.override_settings["sd_model_checkpoint"] = info.name
            p.sd_model_name = info.name
            opts.sd_model_checkpoint = ckpt
            print("「Stable Diffusion 模型」将切换为：", opts.sd_model_checkpoint)
            print("-" * terminal_width)
            print("加载SD模型：", info.name)
            print("-" * terminal_width)
            # shared.sd_model = sd_models.load_model(info)
            sd_models.load_model(info)
            # sd_models.reload_model_weights()
            print("=" * terminal_width)
            print("\n")
