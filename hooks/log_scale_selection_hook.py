# hooks/log_scale_selection_hook.py

from mmengine.hooks import Hook
# from mmengine.runner.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class LogScaleSelectionHook(Hook):
    def after_train_epoch(self, runner):
        head = runner.model.bbox_head
        if hasattr(head, 'log_scale_selection'):
            runner.logger.info("Logging scale selection...")
            head.log_scale_selection(runner.logger)

