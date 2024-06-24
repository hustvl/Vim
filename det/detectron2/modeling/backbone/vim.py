import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous

from .backbone import Backbone
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)

from .vit import SimpleFeaturePyramid
# add the root path to the system path
import sys, os
# import the parent directory of the current cwd
sys.path.append(os.path.dirname(os.getcwd()) + "/vim")
from models_mamba import VisionMamba, layer_norm_fn, rms_norm_fn, RMSNorm
from utils import interpolate_pos_embed

logger = logging.getLogger(__name__)


__all__ = ["VisionMambaDet", "SimpleFeaturePyramid", "get_vim_lr_decay_rate"]


class VisionMambaDet(VisionMamba, Backbone):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=192,
        depth=24,
        use_checkpoint=False, 
        pretrained=None,
        if_fpn=True,
        last_layer_process="none",
        out_feature="last_feat",
        **kwargs,
    ):

        # for rope
        ft_seq_len = img_size // patch_size
        kwargs['ft_seq_len'] = ft_seq_len

        super().__init__(img_size, patch_size, depth=depth, embed_dim=embed_dim, channels=in_chans, num_classes=num_classes, **kwargs)

        self.use_checkpoint = use_checkpoint
        self.if_fpn = if_fpn
        self.last_layer_process = last_layer_process

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        # remove cls head
        del self.head
        del self.norm_f

        self.init_weights(pretrained)

        # drop the pos embed for class token
        if self.if_cls_token:
            del self.cls_token
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])
    

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = logging.getLogger(__name__)

            state_dict = torch.load(pretrained, map_location="cpu")
            state_dict_model = state_dict["model"]
            state_dict_model.pop("head.weight")
            state_dict_model.pop("head.bias")
            # pop rope
            state_dict_model.pop("rope.freqs_cos")
            state_dict_model.pop("rope.freqs_sin")

            if self.patch_embed.patch_size[-1] != state_dict["model"]["patch_embed.proj.weight"].shape[-1]:
                state_dict_model.pop("patch_embed.proj.weight")
                state_dict_model.pop("patch_embed.proj.bias")
            interpolate_pos_embed(self, state_dict_model)

            res = self.load_state_dict(state_dict_model, strict=False) 
            logger.info(res)
            print(res)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def forward_features(self, x, inference_params=None):
        B, C, H, W = x.shape
        # x, (Hp, Wp) = self.patch_embed(x)
        x = self.patch_embed(x)

        batch_size, seq_len, _ = x.size()
        Hp = Wp = int(math.sqrt(seq_len))

        if self.pos_embed is not None:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        features = []
        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # todo: configure this
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if self.last_layer_process == 'none':
            residual = hidden_states
        elif self.last_layer_process == 'add':
            residual = hidden_states + residual
        elif self.last_layer_process == 'add & norm':
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            residual = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
 
        outputs = {self._out_features[0]: residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()}

        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def get_vim_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=24):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".layers." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
