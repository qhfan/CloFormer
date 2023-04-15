from .CloFormer import CloFormer

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'cloformer':
        model = CloFormer(
            in_chans=config.MODEL.cloformer.in_chans,
            num_classes=config.MODEL.cloformer.num_classes,
            embed_dims=config.MODEL.cloformer.embed_dims,
            depths=config.MODEL.cloformer.depths,
            num_heads=config.MODEL.cloformer.num_heads,
            group_splits=config.MODEL.cloformer.group_splits,
            kernel_sizes=config.MODEL.cloformer.kernel_sizes,
            window_sizes=config.MODEL.cloformer.window_sizes,
            mlp_kernel_sizes=config.MODEL.cloformer.mlp_kernel_sizes,
            mlp_ratios=config.MODEL.cloformer.mlp_ratios,
            attn_drop=config.MODEL.cloformer.attn_drop,
            mlp_drop=config.MODEL.cloformer.mlp_drop,
            qkv_bias=config.MODEL.cloformer.qkv_bias,
            drop_path_rate=config.MODEL.cloformer.drop_path_rate,
            use_checkpoint=config.MODEL.cloformer.use_checkpoint
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model