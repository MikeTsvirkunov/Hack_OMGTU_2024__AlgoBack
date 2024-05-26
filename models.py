from typing import Dict, List
from pydantic import BaseModel
import torch


class PrepearedVectorForDB(BaseModel):
    emb_parted: List[List[float]]
    emb_id_parted: List[str]
    emb_meta_parted: List[Dict]


class SingleLinkModel(BaseModel):
    link: str


class DifferentLinksModel(BaseModel):
    links: List[str]