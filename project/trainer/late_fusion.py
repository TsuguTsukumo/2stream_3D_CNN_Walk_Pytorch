try:
    from project.models.make_model import late_fusion
    from project.trainer.base_paired_video import PairedVideoTrainer
except ModuleNotFoundError:
    from models.make_model import late_fusion
    from trainer.base_paired_video import PairedVideoTrainer


class LateFusionTrainer(PairedVideoTrainer):
    def build_model(self, hparams):
        return late_fusion(hparams)
        
