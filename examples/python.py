#!/usr/bin/env python3


# Attempt to import the PulseClient from pulse_sdk.
# If pulse-sdk is not installed or the class name is different,
# this will raise an ImportError.
from pulse.auth import ClientCredentialsAuth
from pulse.core.client import CoreClient
from pulse.dsl import Workflow
from pulse.starters import get_strings


def main():
    texts = get_strings("/home/will/Documents/Test Data/disney-1k.txt")

    # Define lifecycle callbacks
    def on_run_start():
        print("Workflow starting")

    def on_process_start(process_id):
        print(f"Starting process: {process_id}")

    def on_process_end(process_id, result):
        print(f"Finished process: {process_id}, result: {result}")

    def on_run_end():
        print("Workflow finished")

    wf = (
        Workflow()
        .monitor(
            on_run_start=on_run_start,
            on_process_start=on_process_start,
            on_process_end=on_process_end,
            on_run_end=on_run_end,
        )
        .source("comments", texts)
        .theme_generation(min_themes=6, max_themes=20, source="comments")
        .theme_allocation(inputs="comments", themes_from="theme_generation")
        .sentiment(source="comments")
        .cluster(source="comments")
    )

    client = CoreClient(
        auth=ClientCredentialsAuth(
            client_id="NmH4QIb7mYAAGOIqulIM1Vqkes8ipj59",
            client_secret="zUOUrpnYNztJwb50rbC-EwIdxReQ0SzmJu7JumdvrBb5E0pKgBT0zcKNYlMbVLhl",
        )
    )
    results = wf.run(client=client)
    print("Detected themes:", results.theme_generation.themes)
    df_alloc = results.theme_allocation.to_dataframe()
    print(df_alloc)


if __name__ == "__main__":
    main()
