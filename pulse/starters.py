import os
from typing import List, Union
import pandas as pd
from typing import Optional
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import SentimentProcess, ThemeAllocation
from pulse.analysis.results import ThemeAllocationResult


def _load_text(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_csv_tsv(path: str) -> List[str]:
    sep = "," if path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_excel(path: str) -> List[str]:
    df = pd.read_excel(path, sheet_name=0, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


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


def sentiment_analysis(input_data: Union[List[str], str]) -> List[float]:
    """
    Return a polarity score (-1 to 1) for each string.
    Uses TextBlob under the hood.
    """
    texts = get_strings(input_data)

    # Initialize Analyzer with a SentimentProcess instance (not the class)
    analyzer = Analyzer(processes=[SentimentProcess()], dataset=texts)

    resp = analyzer.run()

    return resp.sentiment.sentiments


def theme_allocation(
    input_data: Union[List[str], str], themes: Optional[List[str]] = None
) -> ThemeAllocationResult:
    """
    Allocate each text to one or more themes.
    If `themes` is a list of strings, use those as seed themes.
    If `themes` is None, automatically generate themes via Analyzer and ThemeGeneration.
    Returns a dict mapping  theme_name to list of matched texts.
    """
    texts = get_strings(input_data)

    # 1) Generate or seed themes
    analyzer = Analyzer(processes=[ThemeAllocation(themes=themes)], dataset=texts)

    resp = analyzer.run()

    return resp.theme_allocation
