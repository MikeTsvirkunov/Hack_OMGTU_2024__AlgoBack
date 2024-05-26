from dataclasses import dataclass, field
from typing import Dict
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
from transformers import MarkupLMFeatureExtractor, MarkupLMTokenizerFast, MarkupLMProcessor, MarkupLMModel
import chromadb


@dataclass(frozen=True)
class DataBase:
    client = chromadb.Client()
    collection = client.create_collection(name="links_dataset")


@dataclass(frozen=True)
class RequestsParams:
    header: Dict[str, str] = field(default_factory=lambda: {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'})


@dataclass(frozen=True)
class Models:
    html_vectorization_window: int = 1000
    umap: UMAP = UMAP(n_neighbors=15, min_dist=0.00000001, n_components=2, metric = "cosine")
    aglomerative_model: AgglomerativeClustering = AgglomerativeClustering(n_clusters=2)
    html_tokenizer: MarkupLMProcessor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    html_vectorizer: MarkupLMModel = MarkupLMModel.from_pretrained("microsoft/markuplm-base")


__all__ = [
    RequestsParams,
    Models,
    DataBase
]