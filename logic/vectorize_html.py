import gc
import bs4
import torch
from tqdm import tqdm
from logic.parsers import parse_site_by_requests, remove_context_from_html
from constants import RequestsParams, Models


@torch.no_grad()
def vectorize_html(html_code: str) -> torch.Tensor:
    window = 1000
    r = bs4.BeautifulSoup(remove_context_from_html(html_code), 'html.parser')
    r = r.prettify()
    l = r.split('\n')
    r_p = [''.join(l[i:i+window]) for i in range(0, len(l), window)]
    k = list(map(lambda a: Models.html_vectorizer(**Models.html_tokenizer(a, return_tensors="pt")).last_hidden_state.squeeze(), r_p))
    outputs = torch.cat(k, 0)
    return outputs


def vectorize_html_pipeline(link: str) -> torch.Tensor:
    html_code = parse_site_by_requests(link)
    html_vectors = vectorize_html(html_code)
    return html_vectors
