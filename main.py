from lightning.pytorch.cli import LightningCLI
import torch
import torch.nn.functional as F
import lightning as L
from model import LitTokenClassification
from datamodule import TokenClassificationData


class MyLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.pretrained_model_name",
                              "data.pretrained_model_name")
        parser.link_arguments("model.batch_size", "data.batch_size")
        # parser.link_arguments("data.label_names",
        #                       "model.label_names",
        #                       apply_on="instantiate")


def cli_main():
    cli = MyLightningCLI(
        model_class=LitTokenClassification,
        datamodule_class=TokenClassificationData,
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
