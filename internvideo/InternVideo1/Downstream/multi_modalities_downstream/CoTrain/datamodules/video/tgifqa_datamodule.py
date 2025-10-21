# Modified by Sam Pollard
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datasets import TGIFQADataset
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.datamodule_base import BaseDataModule


class TGIFQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return TGIFQADataset

    @property
    def dataset_cls_no_false(self):
        return TGIFQADataset

    @property
    def dataset_name(self):
        return "tgifqa"
