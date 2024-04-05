import typer

import extract_pair_features

app = typer.Typer()


@app.command()
def extract_raw_packet_features(data_folder: str, dataset_name: str) -> None:
    """
    Extracts packet times and volume features from raw .pcaps to be .

    Args:
        data_folder (str): The path to raw .pcaps dataset.
        dataset_name (str): The name to store the extracted features 
        as csv files into /mnt/nas-shared/torpedo/extracted_features/extracted_features_{dataset_name}/.

    Example:
        $ python3 app.py /mnt/nas-shared/torpedo/new_datasets/OSTest/experiment_results/ OSTest
    """
    typer.echo("Extracting packet times and volume features from raw .pcaps ...")
    extract_pair_features.extract_pairs_features(data_folder, dataset_name)


if __name__ == "__main__":
    app()