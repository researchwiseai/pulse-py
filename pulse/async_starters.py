"""Async starter functions for common Pulse API operations."""

import asyncio
import os
from typing import List, Union, Optional
import pandas as pd
from pulse.analysis.async_analyzer import AsyncAnalyzer
from pulse.analysis.processes import ThemeAllocation
from pulse.analysis.results import ThemeAllocationResult
from pulse.auth import _BaseOAuth2Auth
from pulse.core.async_client import AsyncCoreClient
from pulse.core.async_jobs import AsyncJob
from pulse.core.models import (
    ClusteringResponse,
    SentimentResponse,
    SummariesResponse,
    ThemesResponse,
    ThemeSetsResponse,
    ExtractionsResponse,
    SimilarityResponse,
    UsageEstimateResponse,
)
from pulse.core.async_error_handling import (
    AsyncCancellationError,
    async_timeout_context,
    handle_async_http_errors,
)


def _load_csv_tsv(path: str) -> List[str]:
    """Load strings from CSV or TSV file."""
    sep = "," if path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_excel(path: str) -> List[str]:
    """Load strings from Excel file."""
    df = pd.read_excel(path, sheet_name=0, header=None)
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _load_text(path: str) -> List[str]:
    """Load strings from text file."""
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


@handle_async_http_errors
async def generate_themes_async(
    input_data: Union[List[str], str],
    *,
    min_themes: Optional[int] = None,
    max_themes: Optional[int] = None,
    context: Optional[str] = None,
    version: Optional[str] = None,
    prune: Optional[int] = None,
    interactive: Optional[bool] = None,
    initial_sets: Optional[int] = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
    timeout: Optional[float] = None,
) -> Union[ThemesResponse, ThemeSetsResponse, AsyncJob]:
    """Generate themes using async core client with enhanced error handling.

    Args:
        input_data: List of strings or a path to load strings from.
        min_themes: Minimum number of themes to generate.
        max_themes: Maximum number of themes to generate.
        context: Context string to guide theme generation.
        version: Optional model version for reproducible output.
        prune: Number of themes to prune during generation.
        interactive: Enable interactive theme generation.
        initial_sets: Number of initial theme sets to generate.
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.
        timeout: Optional timeout for the operation in seconds.

    Returns:
        ThemesResponse, ThemeSetsResponse, or AsyncJob handle.

    Raises:
        AsyncTimeoutError: If operation times out.
        AsyncCancellationError: If operation is cancelled.
        PulseAPIError: If API request fails.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    # Calculate appropriate timeout based on data size and mode
    if timeout is None:
        if fast:
            timeout = 60.0  # 1 minute for fast mode
        else:
            timeout = max(300.0, len(texts) * 0.5)  # Scale with input size

    try:
        async with async_timeout_context(
            timeout,
            "generate_themes_async",
            {"text_count": len(texts), "fast_mode": fast},
        ):

            async def generate_themes_operation():
                if client is None:
                    async_client = AsyncCoreClient(auth=auth)
                    async with async_client:
                        return await async_client.generate_themes(
                            texts,
                            min_themes=min_themes,
                            max_themes=max_themes,
                            context=context,
                            version=version,
                            prune=prune,
                            interactive=interactive,
                            initial_sets=initial_sets,
                            fast=fast,
                            await_job_result=await_job_result,
                        )
                else:
                    return await client.generate_themes(
                        texts,
                        min_themes=min_themes,
                        max_themes=max_themes,
                        context=context,
                        version=version,
                        prune=prune,
                        interactive=interactive,
                        initial_sets=initial_sets,
                        fast=fast,
                        await_job_result=await_job_result,
                    )

            return await asyncio.wait_for(generate_themes_operation(), timeout=timeout)
    except asyncio.CancelledError as e:
        raise AsyncCancellationError(
            "Generate themes operation was cancelled",
            operation="generate_themes_async",
            context={"text_count": len(texts), "fast_mode": fast},
        ) from e


async def sentiment_analysis_async(
    input_data: Union[List[str], str],
    *,
    version: str | None = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
) -> Union[SentimentResponse, AsyncJob]:
    """Perform sentiment analysis on input data using the async core client.

    Args:
        input_data: List of strings or a path to load strings from.
        version: Optional model version for reproducible output.
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.

    Returns:
        SentimentResponse or AsyncJob handle.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            return await client.analyze_sentiment(
                texts,
                version=version,
                fast=fast,
                await_job_result=await_job_result,
            )
    else:
        return await client.analyze_sentiment(
            texts,
            version=version,
            fast=fast,
            await_job_result=await_job_result,
        )


async def theme_allocation_async(
    input_data: Union[List[str], str],
    auth: _BaseOAuth2Auth | None = None,
    themes: Optional[List[str]] = None,
    client: Optional[AsyncCoreClient] = None,
) -> ThemeAllocationResult:
    """
    Allocate each text to one or more themes asynchronously.

    If `themes` is a list of strings, use those as seed themes.
    If `themes` is None, automatically generate themes via AsyncAnalyzer and
    ThemeGeneration.

    Args:
        input_data: List of strings or a path to load strings from.
        auth: Optional authentication object.
        themes: Optional list of theme strings to use for allocation.
        client: Existing AsyncCoreClient instance.

    Returns:
        ThemeAllocationResult object.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            async with AsyncAnalyzer(
                processes=[ThemeAllocation(themes=themes)],
                dataset=texts,
                client=client,
                fast=fast,
                auth=auth,
            ) as analyzer:
                resp = await analyzer.run()
                return resp.theme_allocation
    else:
        async with AsyncAnalyzer(
            processes=[ThemeAllocation(themes=themes)],
            dataset=texts,
            client=client,
            fast=fast,
            auth=auth,
        ) as analyzer:
            resp = await analyzer.run()
            return resp.theme_allocation


async def compare_similarity_async(
    input_data: Union[List[str], str],
    *,
    set_a: Optional[List[str]] = None,
    set_b: Optional[List[str]] = None,
    split: Optional[dict] = None,
    flatten: bool = False,
    version: Optional[str] = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
) -> Union[SimilarityResponse, AsyncJob]:
    """Compare similarity between texts with optional splitting support.

    Args:
        input_data: List of strings or a path to load strings from
            (for self-similarity).
        set_a: First set of texts for cross-similarity computation.
        set_b: Second set of texts for cross-similarity computation.
        split: Text splitting configuration for fine-grained analysis.
        flatten: Return flattened results instead of matrix format.
        version: Optional model version for reproducible output.
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.

    Returns:
        SimilarityResponse or AsyncJob handle.
    """

    async def _perform_similarity(client_to_use):
        if set_a is not None and set_b is not None:
            # Cross-similarity mode
            fast = len(set_a) * len(set_b) <= 20_000
            return await client_to_use.compare_similarity(
                set_a=set_a,
                set_b=set_b,
                split=split,
                flatten=flatten,
                version=version,
                fast=fast,
                await_job_result=await_job_result,
            )
        else:
            # Self-similarity mode
            texts = get_strings(input_data)
            fast = len(texts) <= 500
            return await client_to_use.compare_similarity(
                set=texts,
                split=split,
                flatten=flatten,
                version=version,
                fast=fast,
                await_job_result=await_job_result,
            )

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            return await _perform_similarity(client)
    else:
        return await _perform_similarity(client)


async def cluster_analysis_async(
    input_data: Union[List[str], str],
    *,
    k: int,
    algorithm: str = "kmeans",
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
) -> Union[ClusteringResponse, AsyncJob]:
    """Cluster input texts using the `/clustering` endpoint asynchronously.

    Args:
        input_data: List of strings or a path to load strings from.
        k: Desired number of clusters.
        algorithm: Clustering algorithm. Options: "kmeans", "skmeans",
            "agglomerative", "hdbscan". Defaults to "kmeans".
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.

    Returns:
        ClusteringResponse or AsyncJob handle.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 500

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            return await client.cluster_texts(
                inputs=texts,
                k=k,
                algorithm=algorithm,
                fast=fast,
                await_job_result=await_job_result,
            )
    else:
        return await client.cluster_texts(
            inputs=texts,
            k=k,
            algorithm=algorithm,
            fast=fast,
            await_job_result=await_job_result,
        )


async def extract_elements_async(
    input_data: Union[List[str], str],
    dictionary: List[str],
    *,
    type: str = "named-entities",
    expand_dictionary: bool = False,
    expand_dictionary_limit: Optional[int] = None,
    version: Optional[str] = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
) -> Union[ExtractionsResponse, AsyncJob]:
    """Extract elements matching dictionary terms from input texts asynchronously.

    Args:
        input_data: List of strings or a path to load strings from.
        dictionary: List of terms to extract from texts.
        type: Extraction type. Options: "named-entities", "themes".
        expand_dictionary: Expand dictionary entries with synonyms.
        expand_dictionary_limit: Limit for dictionary expansions.
        version: Optional model version for reproducible output.
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.

    Returns:
        ExtractionsResponse or AsyncJob handle.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            return await client.extract_elements(
                inputs=texts,
                dictionary=dictionary,
                type=type,
                expand_dictionary=expand_dictionary,
                expand_dictionary_limit=expand_dictionary_limit,
                version=version,
                fast=fast,
                await_job_result=await_job_result,
            )
    else:
        return await client.extract_elements(
            inputs=texts,
            dictionary=dictionary,
            type=type,
            expand_dictionary=expand_dictionary,
            expand_dictionary_limit=expand_dictionary_limit,
            version=version,
            fast=fast,
            await_job_result=await_job_result,
        )


async def summarize_async(
    input_data: Union[List[str], str],
    question: str,
    *,
    length: str | None = None,
    preset: str | None = None,
    await_job_result: bool = True,
    auth: _BaseOAuth2Auth | None = None,
    client: Optional[AsyncCoreClient] = None,
) -> Union[SummariesResponse, AsyncJob]:
    """Generate a summary of the provided texts asynchronously.

    Args:
        input_data: List of strings or a file path to load from.
        question: Prompt describing what to summarize.
        length: Optional summary length.
        preset: Optional output preset.
        await_job_result: When False, return an AsyncJob handle instead of waiting.
        auth: Optional authentication object.
        client: Existing AsyncCoreClient instance.

    Returns:
        SummariesResponse or AsyncJob handle.
    """
    texts = get_strings(input_data)
    fast = len(texts) <= 200

    if client is None:
        client = AsyncCoreClient(auth=auth)
        async with client:
            return await client.generate_summary(
                texts,
                question,
                length=length,
                preset=preset,
                fast=fast,
                await_job_result=await_job_result,
            )
    else:
        return await client.generate_summary(
            texts,
            question,
            length=length,
            preset=preset,
            fast=fast,
            await_job_result=await_job_result,
        )


async def estimate_usage_async(
    feature: str,
    input_data: Union[List[str], str],
    *,
    client: Optional[AsyncCoreClient] = None,
) -> UsageEstimateResponse:
    """Estimate credit usage for a feature without authentication asynchronously.

    Args:
        feature: Feature to estimate usage for. Options: "embeddings",
            "sentiment", "themes", "extractions", "summaries".
        input_data: List of strings or a path to load strings from.
        client: Existing AsyncCoreClient instance.

    Returns:
        UsageEstimateResponse with estimated usage information.
    """
    texts = get_strings(input_data)

    if client is None:
        client = AsyncCoreClient()
        async with client:
            return await client.estimate_usage(
                feature=feature,
                inputs=texts,
            )
    else:
        return await client.estimate_usage(
            feature=feature,
            inputs=texts,
        )
