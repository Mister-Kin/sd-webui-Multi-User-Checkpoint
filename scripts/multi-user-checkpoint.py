import gradio as gr
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import modules.shared as shared

from modules import scripts
from modules.ui import refresh_symbol
from modules.ui_components import ToolButton
from modules.sd_models import select_checkpoint, load_model
from modules.shared_items import sd_vae_items, refresh_vae_list
from modules.shared import opts, list_checkpoint_tiles, refresh_checkpoints
from modules.sd_models import get_closet_checkpoint_match

class MultiUserCKPT(scripts.Script):
    def title(self):
        return 'Multi User Checkpoint'
    
    def describe(self):
        return "Allow multiple clients to create task queues with different checkpoints."
    
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
                label="SD Checkpoint for you",
                # value="Do not change",
                # choices=["Do not change"]+list_checkpoint_tiles(),
                value=ckpt_choices[0],
                choices=ckpt_choices,
                interactive=True
            )
            self.refresh_ckpt = ToolButton(value=refresh_symbol, elem_id="muc_refresh_ckpt")
            self.refresh_ckpt.click(
                fn=self.refresh_ckpts,
                inputs=[],
                outputs=[self.checkpoint]
            )
            
            vae_choices=sd_vae_items()
            self.vae = gr.Dropdown(
                elem_id="muc_vae",
                label="SD VAE for you",
                # value="Do not change",
                # choices=["Do not change"]+sd_vae_items(),
                value=vae_choices[0],
                choices=vae_choices,
                interactive=True
            )
            self.refresh_vae = ToolButton(value=refresh_symbol, elem_id="muc_refresh_vae")
            self.refresh_vae.click(
                fn=self.refresh_vaes,
                inputs=[],
                outputs=[self.vae]
            )
    
    def process(self, p, ckpt:str, vae:str):
        # if vae != opts.sd_vae and vae != "Do not change":
        #     opts.sd_vae = vae
        # if ckpt != opts.sd_model_checkpoint and ckpt != "Do not change":
        #     opts.sd_model_checkpoint = ckpt
        #     load_model(checkpoint_info=select_checkpoint())

        if vae != opts.sd_vae:
            print("原 sd_vae: ", opts.sd_vae)
            opts.sd_vae = vae
            p.sd_vae_name = vae
            p.override_settings['sd_vae'] = vae
            sd_vae.reload_vae_weights()
            print("切换后 sd_vae:", opts.sd_vae)
        if ckpt != opts.sd_model_checkpoint:
            print("原 sd_model_checkpoint:", opts.sd_model_checkpoint)
            info = get_closet_checkpoint_match(ckpt)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {ckpt}")
            p.override_settings['sd_model_checkpoint'] = info.name
            p.sd_model_name = info.name
            opts.sd_model_checkpoint = ckpt
            print("加载模型：", info.name)
            # shared.sd_model = sd_models.load_model(info)
            sd_models.load_model(info)
            # sd_models.reload_model_weights()
            print("切换后 sd_model_checkpoint:", opts.sd_model_checkpoint)

            