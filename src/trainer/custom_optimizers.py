from transformers.optimization import Adafactor, AdafactorSchedule


def get_adafactor_os(model):
    optimizer = Adafactor(
        model.parameters(),
        sclae_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None,
    )

    lr_scheduler = AdafactorSchedule(optimizer)

    return optimizer, lr_scheduler
