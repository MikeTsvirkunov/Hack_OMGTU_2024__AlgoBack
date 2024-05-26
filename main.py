import base64
import datetime
import io
from typing import Union
import uuid

from fastapi.responses import FileResponse, JSONResponse
from matplotlib import pyplot as plt
import numpy as np
from constants import DataBase
from logic.parsers import parse_site_by_requests
from logic.pipelines import clusterization_pipeline
from logic.vector_processor import insert_in_database, prep_emb_for_database
from logic.vectorize_html import vectorize_html, vectorize_html_pipeline
from models import SingleLinkModel, DifferentLinksModel

from fastapi import FastAPI, HTTPException


app = FastAPI()


@app.get("/check_run")
def read_root():
    return {"Hello": "World"}


@app.post("/add_link")
async def read_item(data: SingleLinkModel):
    v = vectorize_html_pipeline(data.link)
    v_prep = prep_emb_for_database(v)
    insert_in_database(v_prep)
    return {
        'state': 'success', 
        'link': data.link,
        'ids': v_prep.emb_id_parted
    }


@app.post("/add_links")
async def read_item(data: DifferentLinksModel):
    ids = []
    for i in data.links:
        v = vectorize_html_pipeline(i)
        v_prep = prep_emb_for_database(v)
        ids.append(v_prep.emb_id_parted)
        insert_in_database(v_prep)
    return {
        'state': 'success',
        'link_ids_pairs': dict(zip(data.links, ids))
    }


@app.post("/compare_links/{method}{req_img}")
async def read_item(method: str, req_img: int, data: DifferentLinksModel):
    list_of_vectors = []
    for link in data.links:
        if method == 'embeding':
            html_code = parse_site_by_requests(link)
            html_vectors = vectorize_html(html_code)
            list_of_vectors.append(
                html_vectors.mean(dim=0).detach().numpy()
            )
        elif method == 'sequential':
            html_code = parse_site_by_requests(link)
            html_vectors = vectorize_html(html_code)
            list_of_vectors += html_vectors.detach().numpy().tolist()
        else:
            raise HTTPException(status_code=404, detail="Method not found")
    
    r, c = clusterization_pipeline(list_of_vectors)
    r = np.array(r)
    my_stringIObytes = io.BytesIO()
    plt.scatter(r[:, 0], r[:, 1], c)
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    return JSONResponse(
        content={
            'clusted_urls': {l_i: c_i  for l_i, c_i in zip(data.links, c.tolist())},
            'scatter': my_base64_jpgData
        } if bool(req_img) else {
            'clusted_urls': {l_i: c_i  for l_i, c_i in zip(data.links, c.tolist())},
        }
    )

