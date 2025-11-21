import os
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


i2v_14B = EasyDict(__name__='Config: Wan I2V 14B')
i2v_14B.update(wan_shared_cfg)

i2v_14B.base_dir = make_abs_path('../pretrained_weights/Wan2.1-I2V-14B-720P')

i2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_14B.clip_dtype = torch.float16
i2v_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
i2v_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
i2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_14B.vae_stride = (4, 8, 8)

# transformer
i2v_14B.patch_size = (1, 2, 2)
i2v_14B.dim = 5120
i2v_14B.ffn_dim = 13824
i2v_14B.freq_dim = 256
i2v_14B.num_heads = 40
i2v_14B.num_layers = 40
i2v_14B.window_size = (-1, -1)
i2v_14B.qk_norm = True
i2v_14B.cross_attn_norm = True
i2v_14B.eps = 1e-6

i2v_14B.prompt = "A person is talking."
i2v_14B.neg_prompt = "The camera and shot flickers, the camera and shot is discontinuous, there is a cut, the face is not clear, the tone is too bright, overexposed, still, the details are blurry, the subtitles, the style, the work, the painting, the picture, still, the overall is grayish, the worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces Deformed, disfigured, deformed limbs, fingers fused, a still picture, a messy background, three legs, many people in the background, walking backwards"
