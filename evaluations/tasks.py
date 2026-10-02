"""Benchmark task lists and modality groupings used in the paper."""
from __future__ import annotations

# MTEB-v2: 8 BEIR retrieval tasks (nDCG@10)
MTEB_TASKS = [
    "ArguAna", "CQADupstackEnglishRetrieval", "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval", "FiQA2018", "NFCorpus", "SCIDOCS", "SciFact",
]

# MAEB: 22-task English subset, split into speech (12) and general audio (10)
MAEB_SPEECH_TASKS = [
    "GigaSpeechT2ARetrieval", "SpokenSQuADT2ARetrieval", "CREMA_D",
    "CommonLanguageAgeDetection", "IEMOCAPGender", "VoxCelebSA",
    "CREMA_DClustering", "CREMADPairClassification", "NMSQAPairClassification",
    "VoxPopuliAccentPairClassification", "RavdessZeroshot",
    "SpeechCommandsZeroshotv0.02",
]
MAEB_AUDIO_TASKS = [
    "ClothoT2ARetrieval", "MACST2ARetrieval", "UrbanSound8KT2ARetrieval",
    "BeijingOpera", "BirdCLEF", "GTZANGenre", "MridinghamTonic",
    "VehicleSoundClustering", "FSD2019Kaggle", "GTZANAudioReranking",
]
MAEB_TASKS = MAEB_SPEECH_TASKS + MAEB_AUDIO_TASKS

# ViDoRe-V3: 7 English visually-rich document retrieval tasks (nDCG@10)
VIDORE_TASKS = [
    "Vidore3FinanceEnRetrieval.v2", "Vidore3HrRetrieval.v2",
    "Vidore3IndustrialRetrieval.v2", "Vidore3PharmaceuticalsRetrieval.v2",
    "Vidore3ComputerScienceRetrieval.v2", "Vidore3EnergyRetrieval.v2",
    "Vidore3PhysicsRetrieval.v2",
]
VIDORE_LANGUAGES = ["eng-Latn"]

# MMEB-V2: 10 image + 6 video tasks (hit@1)
MMEB_IMAGE_TASKS = [
    "VOC2007", "Country211", "OK-VQA", "EDIS", "MSCOCO_t2i", "VisualNews_t2i",
    "MSCOCO_i2t", "VisualNews_i2t", "NIGHTS", "FashionIQ",
]
MMEB_VIDEO_TASKS = ["HMDB51", "UCF101", "MSR-VTT", "MSVD", "DiDeMo", "VATEX"]
MMEB_TASKS = MMEB_IMAGE_TASKS + MMEB_VIDEO_TASKS

BENCHMARKS = {
    "mteb": MTEB_TASKS,
    "maeb": MAEB_TASKS,
    "mmeb": MMEB_TASKS,
    "vidore": VIDORE_TASKS,
}

# (modality, benchmark, tasks) in the order reported in the paper
MODALITIES = [
    ("text", "mteb", MTEB_TASKS),
    ("speech", "maeb", MAEB_SPEECH_TASKS),
    ("audio", "maeb", MAEB_AUDIO_TASKS),
    ("image", "mmeb", MMEB_IMAGE_TASKS),
    ("video", "mmeb", MMEB_VIDEO_TASKS),
    ("visdoc", "vidore", VIDORE_TASKS),
]
