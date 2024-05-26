from typing import Tuple
import uuid
import torch

from constants import DataBase
from models import PrepearedVectorForDB


def prep_emb_for_database(v: torch.Tensor) -> PrepearedVectorForDB:
    emb_parted = []
    emb_id_parted = []
    emb_meta_parted = []
    v_id = str(uuid.uuid4())
    for i, v_i in enumerate(v):
        emb_parted.append(v_i.numpy().tolist())
        emb_id_parted.append(v_id + '_' + str(i))
        emb_meta_parted.append({'parent_id': v_id, 'index': i, 'seq_len': v.shape[0]})
    return PrepearedVectorForDB(
        emb_id_parted=emb_id_parted,
        emb_meta_parted=emb_meta_parted,
        emb_parted=emb_parted
    )



def insert_in_database(v_prep: PrepearedVectorForDB):
    DataBase.collection.add(
        embeddings=v_prep.emb_parted,
        ids=v_prep.emb_id_parted,
        metadatas=v_prep.emb_meta_parted
    )