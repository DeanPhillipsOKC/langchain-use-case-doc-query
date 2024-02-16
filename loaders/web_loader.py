from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

def get_loader(url: str) -> WebBaseLoader:
    return WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )