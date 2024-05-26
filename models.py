from typing import List
from pydantic import BaseModel


class SingleLinkModel(BaseModel):
    link: str


class DifferentLinksModel(BaseModel):
    links: List[str]