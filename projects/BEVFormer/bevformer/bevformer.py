from mmdet.models.backbones import ResNet

from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello BEVFormer!')
        super().__init__(**kwargs)

    def forward(self, **kwargs) -> None:
        print('this is forward!')
        super().__init__(**kwargs)
