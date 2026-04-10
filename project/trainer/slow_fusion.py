try:
    from project.models.make_model import slow_fusion
    from project.trainer.base_paired_video import PairedVideoTrainer
except ModuleNotFoundError:
    from models.make_model import slow_fusion
    from trainer.base_paired_video import PairedVideoTrainer


class SlowFusionTrainer(PairedVideoTrainer):
    def build_model(self, hparams):
        return slow_fusion(hparams)
        
