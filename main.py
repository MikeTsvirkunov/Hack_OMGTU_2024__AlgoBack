import datetime
from typing import Union
import uuid
from constants import DataBase
from logic.parsers import parse_site_by_requests
from logic.vectorize_html import vectorize_html, vectorize_html_pipeline
from models import SingleLinkModel, DifferentLinksModel

from fastapi import FastAPI


app = FastAPI()


@app.get("/check_run")
def read_root():
    return {"Hello": "World"}


@app.post("/add_link")
async def read_item(data: SingleLinkModel):
    v = vectorize_html_pipeline(data.link)
    v_id = str(uuid.uuid4())
    DataBase.collection.add(
        documents=[data.link],
        embeddings=[v.numpy().reshape(-1).tolist()],
        metadatas=[{'time': str(datetime.datetime.now()), 'len_of_seq': v.shape[0], 'len_of_embeding': v.shape[1]}],
        ids=[v_id],
    )
    return {'state': 'success', 'id_link_pairs': dict(zip([v_id], [data.links]))}


@app.post("/add_links")
async def read_item(data: DifferentLinksModel):
    vectors = []
    ids = []
    metadates = []
    for i in data.links:
        v = vectorize_html_pipeline(i)
        vectors.append(v.reshape(-1).tolist())
        ids.append(str(uuid.uuid4()))
        metadates.append({'time': str(datetime.datetime.now()), 'len_of_seq': v.shape[0], 'len_of_embeding': v.shape[1]})
    DataBase.collection.add(
        documents=data.links,
        embeddings=vectors,
        metadatas=metadates,
        ids=ids,
    )
    return {'state': 'success', 'id_link_pairs': dict(zip(ids, data.links))}


@app.post("/compare_links/{method}")
async def read_item(method: str, data: DifferentLinksModel):
    for link in data.links:
        if method == 'embeding':

            html_code = parse_site_by_requests(data.link)
            html_vectors = vectorize_html(html_code)


    return {"result": "succes"}

