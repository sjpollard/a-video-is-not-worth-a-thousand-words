# Modified by Sam Pollard
# pretrain dataset
## video
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.webvid_datamodule import WEBVIDDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.webvid10m_datamodule import WEBVID10MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.howto100m_datamodule import HT100MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.youtube_datamodule import YOUTUBEDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.yttemporal_datamodule import YTTemporalMDataModule
## image
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.cc3m_datamodule import CC3MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.cc12m_datamodule import CC12MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.yfcc15m_datamodule import YFCC15MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.laion400m_datamodule import LAION400MDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.vg_caption_datamodule import VisualGenomeCaptionDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.conceptual_caption_datamodule import ConceptualCaptionDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.sbu_datamodule import SBUCaptionDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.mix100m_datamodule import MIX100MDataModule
# finetune dataset
## image
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.vqav2_datamodule import VQAv2DataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.nlvr2_datamodule import NLVR2DataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.vcr_datamodule import VCRDataModule
## video
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.msrvtt_datamodule import MSRVTTDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.msrvttqa_datamodule import MSRVTTQADataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.msrvtt_choice_datamodule import MSRVTTChoiceDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.msvd_datamodule import MSVDDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.msvdqa_datamodule import MSVDQADataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.ego4d_datamodule import Ego4DDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.tvqa_datamodule import TVQADataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.lsmdc_choice_datamodule import LSMDCChoiceDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.ego4d_choice_datamodule import EGO4DChoiceDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.tgif_datamodule import TGIFDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.tgifqa_datamodule import TGIFQADataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.didemo_datamodule import DIDEMODataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.hmdb51_datamodule import HMDB51DataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.ucf101_datamodule import UCF101DataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.k400_datamodule import K400DataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.lsmdc_datamodule import LSMDCDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.video.k400_video_datamodule import K400VideoDataModule
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.activitynet_datamodule import ActivityNetDataModule

_datamodules = {
    # image
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "cc3m": CC3MDataModule,
    "cc12m": CC12MDataModule,
    'yfcc15m': YFCC15MDataModule,
    'laion400m': LAION400MDataModule,
    'vcr': VCRDataModule,
    'mix100m': MIX100MDataModule,
    # video
    'howto100m': HT100MDataModule,
    'youtube': YOUTUBEDataModule,
    'webvid': WEBVIDDataModule,
    'webvid10m': WEBVID10MDataModule,
    'msrvtt': MSRVTTDataModule,
    'msrvttqa': MSRVTTQADataModule,
    'msrvtt_choice': MSRVTTChoiceDataModule,
    'msvd': MSVDDataModule,
    'msvdqa': MSVDQADataModule,
    'ego4d': Ego4DDataModule,
    'tvqa': TVQADataModule,
    'lsmdc_choice': LSMDCChoiceDataModule,
    'ego4d_choice': EGO4DChoiceDataModule,
    'yttemporal': YTTemporalMDataModule,
    'tgif': TGIFDataModule,
    "tgifqa": TGIFQADataModule,
    'didemo': DIDEMODataModule,
    'hmdb51': HMDB51DataModule,
    'ucf101': UCF101DataModule,
    'k400': K400DataModule,
    'lsmdc': LSMDCDataModule,
    'activitynet': ActivityNetDataModule,
    'k400_video': K400VideoDataModule,
}
