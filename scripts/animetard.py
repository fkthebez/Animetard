import torch
import os
import functools
import sys
import gradio as gr

from modules import scripts, ui, devices
import modules.shared as shared
from modules.shared import opts
from modules.textual_inversion.textual_inversion import Embedding
from modules.sd_hijack import model_hijack

k_sentinel = "$ANIMETARD_EMBEDDING"
k_prompt_map = {
   "landscape" : ("masterpiece (absurdres:1.1) (portrait:1.3) wallpaper (landscape background:1.3) intricate (high detail:1.2), soft lighting, ", "",
                  f"sketch by {k_sentinel}, ", ""),
   "portrait" : ("masterpiece (absurdres:1.1) (portrait:1.3) wallpaper intricate (high detail:1.2), soft lighting, ", "",
                 f"sketch by {k_sentinel}, ", ""),
}

k_basedir = scripts.basedir()

g = {
   "enable" : True,
   "adv_waifu" : False,
   "coomer" : False,
   "prompt_tmpl" : next(iter(k_prompt_map)),
}

def ovr_sampler(steps):
   if steps <= 6:
      return "DPM++ 2M", True
   elif 7 <= steps <= 10:
      return "DPM++ 2M Karras", True
   else:
      return "DPM++ 2M Karras", False

def apply_tmpl(s, pre, suf):
   return pre + "\n" + s + ("\n " + suf if suf != "" else "")

def replace_match(v, generic_anime):
   return [x.replace(k_sentinel, ("animetard_bad-artist-anime" if generic_anime else
                                  "animetard_bad-artist")) for x in v]

def ovr_prompt(tmpl, pos, neg, generic_anime, adv_waifu, coomer):
   v = replace_match(k_prompt_map[tmpl], generic_anime)

   return ([apply_tmpl(x, ("(nsfw:1.2) " if coomer else "") + v[0] +
                       ("solo 1girl, " if adv_waifu else ""), v[1]) for x in pos],
           [apply_tmpl(x, v[2], v[3]) for x in neg])

def ovr_cfgs(steps, cfgs):
   # save the user from himself
   if steps <= 6:
      return min(cfgs, 3.5)
   else:
      return min(cfgs, 4.0 + (steps - 7) * 0.5)

def proc(p, _enable, _generic_anime, _adv_waifu, _coomer, _prompt_tmpl, _ovr_steps,
         _ovr_sampler, _ovr_prompt, _ovr_cfgs, _ovr_cfg_dumb, _ovr_clip_skip):
   if not _enable:
      return

   if _ovr_steps:
      p.steps = max(min(p.steps, 15), 5)
   if _ovr_sampler:
      p.sampler_name, opts.always_discard_next_to_last_sigma = ovr_sampler(p.steps)
   if _ovr_prompt:
      p.all_prompts, p.all_negative_prompts = ovr_prompt(_prompt_tmpl, p.all_prompts,
                                                         p.all_negative_prompts,
                                                         _generic_anime, _adv_waifu,
                                                         _coomer)
   if _ovr_cfg_dumb:
      p.cfg_scale = 3.5
   elif _ovr_cfgs:
      p.cfg_scale = ovr_cfgs(p.steps, p.cfg_scale)
   if _ovr_clip_skip:
      opts.CLIP_stop_at_last_layers = 2

def make_ui():
   elms = []
   with gr.Group():
      with gr.Accordion("Animetard", open=True):
         gr.HTML(value="<i>\"Think less, prompt more.\"</i>")

         with gr.Row():
            with gr.Column():
               def update_fn(v, T):
                  def x(elm):
                     g[v] = T(elm)
                  
                  return x

               enable = gr.Checkbox(label="Enable", value=True)
               enable.change(fn=update_fn("enable", bool), inputs=enable)
               elms.append(enable)

               elms.append(gr.Checkbox(label=("Use more generic anime style"),
                                       value=False))

               adv_waifu = gr.Checkbox(label=("Use advanced waifu-generation "
                                              "prompting techniques"), value=False)
               adv_waifu.change(fn=update_fn("adv_waifu", bool), inputs=enable)
               elms.append(adv_waifu)


               coomer = gr.Checkbox(label=("Optimize for coomer"), value=False)
               coomer.change(fn=update_fn("coomer", bool), inputs=enable)
               elms.append(coomer)

               gr.HTML("\n");

               prompt_tmpl = gr.Dropdown(label="Prompt template",
                                         choices=[k for k in k_prompt_map],
                                         value=next(iter(k_prompt_map)))
               prompt_tmpl.change(fn=update_fn("prompt_tmpl", str),
                                  inputs=prompt_tmpl)
               elms.append(prompt_tmpl)

            with gr.Accordion("Super advanced (very scary)", open=False):
               elms.append(gr.Checkbox(label="Override steps", value=True))
               elms.append(gr.Checkbox(label="Override sampler", value=True))
               elms.append(gr.Checkbox(label="Override prompt", value=True))
               elms.append(gr.Checkbox(label="Override CFG scale", value=True))
               elms.append(gr.Checkbox(label=("Always override CFG scale to a good "
                                              "value"), value=True))
               elms.append(gr.Checkbox(label=("Override CLIP-skip"), value=True))

   return elms

def load_embeddings():
   # taken from the main repo
   def process_file(path, filename):
      name, ext = os.path.splitext(filename)
      data = torch.load(path, map_location="cpu")

      param_dict = data["string_to_param"]
      vec = next(iter(param_dict.items()))[1].detach().to(devices.device,
                                                          dtype=torch.float32)

      emb = Embedding(vec, name)
      emb.step = data.get("step", None)
      emb.sd_checkpoint = data.get("sd_checkpoint", None)
      emb.sd_checkpoint_name = data.get("sd_checkpoint_name", None)
      emb.vectors = vec.shape[0]
      emb.shape = vec.shape[-1]

      model_hijack.embedding_db.register_embedding(emb, shared.sd_model)

   for root, dirs, fns in os.walk(os.path.join(k_basedir, "models")):
      for fn in fns:
         full_fn = os.path.join(root, fn)
         if os.stat(full_fn).st_size == 0:
            continue
         
         process_file(full_fn, fn)

def wrap(old_fn, new_fn):
   @functools.wraps(old_fn)
   def x(*args, **kwargs):
      return new_fn(old_fn, *args, **kwargs)

   return x

# override the token counter so it is an accurate representation
def update_token_counter_wrap(fn, text, steps):
   if g["enable"]:
      text = ovr_prompt(g["prompt_tmpl"], [text], [""], False, g["adv_waifu"], g["coomer"])[0][0]

   return fn(text, steps)

def init():
   ui.update_token_counter = wrap(ui.update_token_counter, update_token_counter_wrap)
   load_embeddings()

class Animetard(scripts.Script):
   def __init__(self):
      return init()

   def title(self):
      return "Animetard"

   def show(self, is_img2img):
      return scripts.AlwaysVisible

   def ui(self, is_img2img):
      return make_ui()

   def process(self, *args, **kwargs):
      return proc(*args, **kwargs)
