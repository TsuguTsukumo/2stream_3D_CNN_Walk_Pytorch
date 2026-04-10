try:
    from project.models.make_model import early_fusion
    from project.trainer.base_paired_video import PairedVideoTrainer
except ModuleNotFoundError:
    from models.make_model import early_fusion
    from trainer.base_paired_video import PairedVideoTrainer


class EarlyFusionTrainer(PairedVideoTrainer):
    def build_model(self, hparams):
        return early_fusion(hparams)
        
