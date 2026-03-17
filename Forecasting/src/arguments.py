from omegaconf import OmegaConf
import argparse

def parser():
    parser = argparse.ArgumentParser(description='Time-series representation framework')
    parser.add_argument('--model_name', type=str, help='model name for experiments')
    parser.add_argument('--default_cfg', type=str, help='configuration for default setting')
    parser.add_argument('--model_cfg', type=str, help='configuration for model')    
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load configs
    cfg = OmegaConf.load(args.default_cfg)

    # Support dataset-specific model keys like "PatchTST_ETTh1".
    # In that case, MODEL.modelname is set to the base class name (e.g. "PatchTST")
    # so that create_model() can still find the correct class, while MODELSETTING
    # is populated from the dataset-specific config block.
    model_key = args.model_name          # key used to look up MODELSETTING (e.g. PatchTST_ETTh1)
    base_modelname = args.model_name.split('_')[0]  # class name  (e.g. PatchTST)
    cfg = OmegaConf.merge(cfg, {'MODEL': {'modelname': base_modelname}})

    if args.model_cfg:
        model_cfg = OmegaConf.load(args.model_cfg)

        # First try the full dataset-specific key (e.g. PatchTST_ETTh1),
        # then fall back to the base model name (e.g. PatchTST).
        if model_key in model_cfg:
            model_setting_conf = OmegaConf.create(model_cfg[model_key])
            cfg = OmegaConf.merge(cfg, {'MODELSETTING': model_setting_conf})
        elif base_modelname in model_cfg:
            model_setting_conf = OmegaConf.create(model_cfg[base_modelname])
            cfg = OmegaConf.merge(cfg, {'MODELSETTING': model_setting_conf})
        else:
            print(f"Model '{model_key}' (and base '{base_modelname}') not found in the model_config.")
            return None

    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        try:
            # types except for int, float, and str
            OmegaConf.update(cfg, k, eval(v), merge=True)
        except:
            OmegaConf.update(cfg, k, v, merge=True)

    return cfg  