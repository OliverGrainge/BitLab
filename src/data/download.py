from datasets import load_dataset 


def download_alpaca(): 
    """
    Download the Alpaca dataset from the Hugging Face Hub.
    Returns:
        A Hugging Face Dataset object containing the Alpaca data.
    """
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset


DOWNLOAD_DATASETS_REGISTRY = {
    "alpaca": download_alpaca,
}