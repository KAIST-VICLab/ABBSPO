# hooks/set_loss_weight_hook.py

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class SetLossWeightHook(Hook):
    """A custom hook to set the loss weight of bbox loss at a specific epoch."""

    def __init__(self, epoch, target_weight):
        """
        Args:
            epoch (int): The epoch at which to set the loss weight.
            target_weight (float): The target loss weight to set.
        """
        self.epoch = epoch
        self.target_weight = target_weight

    def before_train_epoch(self, runner):
        """Set the loss weight before a specific epoch."""
        current_epoch = runner.epoch + 1  
        if current_epoch == self.epoch:
            runner.model.bbox_head.loss_bbox.loss_weight = self.target_weight
            runner.logger.info(
                f'Set bbox_head.loss_bbox.loss_weight to {self.target_weight} at epoch {self.epoch}'
            )
