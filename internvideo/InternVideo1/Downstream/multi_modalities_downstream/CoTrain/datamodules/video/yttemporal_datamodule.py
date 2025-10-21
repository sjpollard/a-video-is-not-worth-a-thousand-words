# Modified by Sam Pollard
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datasets import YTTemporalDataset
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.datamodule_base import BaseDataModule


class YTTemporalMDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return YTTemporalDataset

    @property
    def dataset_cls_no_false(self):
        return YTTemporalDataset

    @property
    def dataset_name(self):
        return "yttemporal"
