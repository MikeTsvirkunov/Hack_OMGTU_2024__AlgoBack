import bs4
import requests
from constants import RequestsParams


def parse_site_by_requests(link: str) -> str:
    return requests.get(link, headers=RequestsParams().header).text


def remove_context_from_html(html_code: str) -> str:
    soup = bs4.BeautifulSoup(html_code, 'html.parser')
    for element in soup.find_all(text=True):
        if isinstance(element, bs4.Comment):
            continue
        element.replace_with('')
    for script_or_style in soup(['script', 'style']):
        script_or_style.clear()
    return str(soup.prettify())
