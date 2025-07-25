import os
from typing import List, Union
import pandas as pd
from typing import Optional
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import SentimentProcess, ThemeAllocation
from pulse.analysis.results import SentimentResult, ThemeAllocationResult
from pulse.auth import _BaseOAuth2Auth
from pulse.core.client import CoreClient
from pulse.core.jobs import Job
from pulse.core.models import ClusteringResponse


def _load_csv_tsv(path: str) -> List[str]:
    sep = "," if path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_excel(path: str) -> List[str]:
    df = pd.read_excel(path, sheet_name=0, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]


def get_strings(source: Union[List[str], str]) -> List[str]:
    """
    Load input strings from a list or a file path.
    Supports .txt, .csv, .tsv, .xls, .xlsx
    """
    if isinstance(source, list):
        return source
    if not isinstance(source, str) or not os.path.exists(source):
        raise ValueError("Provide a list of strings or a valid file path")
    ext = os.path.splitext(source)[1].lower()
    if ext == ".txt":
        return _load_text(source)
    if ext in (".csv", ".tsv"):
        return _load_csv_tsv(source)
    if ext in (".xls", ".xlsx"):
        return _load_excel(source)
    raise ValueError(f"Unsupported file type: {ext}")


def sentiment_analysis(
    input_data: Union[List[str], str],
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[CoreClient] = None,
) -> List[SentimentResult]:
    """
    Perform sentiment analysis on input data.
    Returns a list of SentimentResult objects.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    analyzer = Analyzer(
        processes=[SentimentProcess()],
        dataset=texts,
        client=client,
        fast=fast,
        auth=auth,
    )

    resp = analyzer.run()

    return resp.sentiment


def theme_allocation(
    input_data: Union[List[str], str],
    auth: _BaseOAuth2Auth | None = None,
    themes: Optional[List[str]] = None,
    client: Optional[CoreClient] = None,
) -> ThemeAllocationResult:
    """
    Allocate each text to one or more themes.
    If `themes` is a list of strings, use those as seed themes.
    If `themes` is None, automatically generate themes via Analyzer and ThemeGeneration.
    Returns a ThemeAllocationResult object.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    analyzer = Analyzer(
        processes=[ThemeAllocation(themes=themes)],
        dataset=texts,
        client=client,
        fast=fast,
        auth=auth,
    )

    resp = analyzer.run()

    return resp.theme_allocation


def cluster_analysis(
    input_data: Union[List[str], str],
    *,
    k: int,
    algorithm: str | None = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[CoreClient] = None,
) -> Union[ClusteringResponse, Job]:
    """Cluster input texts using the core API."""

    texts = get_strings(input_data)
    fast = len(texts) <= 200

    client = client or CoreClient(auth=auth)

    return client.cluster_texts(
        texts,
        k=k,
        algorithm=algorithm,
        fast=fast,
        await_job_result=await_job_result,
    )
