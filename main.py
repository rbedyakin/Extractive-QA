from lightning.pytorch.cli import LightningCLI
import torch
import torch.nn.functional as F
import lightning as L
from src.model import T5QAModel, ModernBertQAModel # noqa: F401
from src.datamodule import T5QAData, ModernBertQAData # noqa: F401


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.init_args.pretrained_model_name",
                              "data.init_args.pretrained_model_name")
        parser.link_arguments("model.init_args.batch_size", "data.init_args.batch_size")
        # parser.link_arguments("data.label_names",
        #                       "model.label_names",
        #                       apply_on="instantiate")


def cli_main():
    cli = MyLightningCLI(
        trainer_class=L.Trainer,
        #  trainer_defaults={
        #      "max_epochs": 3,
        #      "deterministic": True
        #  },
        run=False)
    # note: don't call fit!!

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        # ckpt_path=cli.config.ckpt_path if cli.config.ckpt_path else None,
    )

    cli.trainer.test(
        model=cli.model,
        datamodule=cli.datamodule,
    )

    # cli.trainer.predict(
    #     model=cli.model,
    #     datamodule=cli.datamodule,
    # )


if __name__ == "__main__":
    cli_main()
