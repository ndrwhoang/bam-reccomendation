import ast


STR_PARAMS = ["pretrained_name"]


def update_pretrained_config(pretrained_config, model_config):
    model_config = dict(model_config)
    for k, v in model_config.items():
        if k not in STR_PARAMS:
            model_config[k] = ast.literal_eval(v)
        else:
            pass
    pretrained_config.update(model_config)

    return pretrained_config
