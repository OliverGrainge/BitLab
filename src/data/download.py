import os
import shutil
from pathlib import Path

from datasets import load_dataset, Dataset, load_from_disk

from src.data import dataset
from src.utils import data_path, get_data_dir


def cleanup_hf_artifacts():
    """
    Remove HuggingFace lock files and other artifacts from the project root.
    These are created by the datasets library during streaming downloads.
    """
    repo_root = Path(__file__).parent.parent.parent
    
    removed = []
    
    # Remove everything inside .locks folder
    locks_dir = repo_root / ".locks"
    if locks_dir.exists() and locks_dir.is_dir():
        try:
            shutil.rmtree(locks_dir)
            removed.append(str(locks_dir))
        except OSError:
            pass  # Directory might have been removed already or doesn't exist
    
    # Remove datasets--HuggingFaceFW* folders
    for hf_folder in repo_root.glob("datasets--*"):
        if hf_folder.is_dir():
            try:
                shutil.rmtree(hf_folder)
                removed.append(str(hf_folder))
            except OSError:
                pass  # Directory might have been removed already or doesn't exist
    
    if removed:
        print(f"Cleaned up {len(removed)} artifact(s) from project root")


def download_alpaca(): 
    """
    Download the Alpaca dataset from the Hugging Face Hub.
    Returns:
        A Hugging Face Dataset object containing the Alpaca data.
    """
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset


def download_fineweb_edu(
    num_samples: int = 1000,#1000000,
    buffer_size: int = 10, #10000,
    seed: int = 0,
):
    """
    Download and save a small subset of FineWeb-Edu to disk.

    Args:
        output_dir: path relative to BITLAB_DATA_DIR (default: fineweb-edu)
        num_samples: number of documents to download
        buffer_size: shuffle buffer for streaming
        seed: RNG seed

    Returns:
        A Hugging Face Dataset object loaded from disk.
    """
    from datasets import load_dataset, Dataset

    output_dir = data_path("fineweb-edu")

    # Avoid multiprocessing issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"Streaming {num_samples} samples from FineWeb-Edu...")
    
    # Stream the dataset
    stream_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True,
    )
    
    # Shuffle and take samples
    stream_ds = stream_ds.shuffle(seed=seed, buffer_size=buffer_size)
    stream_ds = stream_ds.take(num_samples)
    
    # Collect samples into a list
    samples = []
    for sample in stream_ds:
        samples.append(sample)
    
    print(f"Collected {len(samples)} samples")
    
    # Convert to Dataset
    if samples:
        dataset = Dataset.from_dict({
            key: [sample[key] for sample in samples]
            for key in samples[0].keys()
        })
    else:
        raise ValueError("No samples collected!")
    
    # Save to disk
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved! You can now load it with: load_from_disk('{output_dir}')")
    
    # Clean up any artifacts created in project root
    cleanup_hf_artifacts()
    
    return dataset


def download_falcon_refinedweb(
    num_samples: int = 1000,#1000000,
    buffer_size: int = 10,#10000,
    seed: int = 0,
):
    """
    Download and save a subset of Falcon-RefinedWeb to disk.

    Args:
        output_dir: path relative to BITLAB_DATA_DIR (default: falcon-refinedweb)
        num_samples: number of documents to download
        buffer_size: shuffle buffer for streaming
        seed: RNG seed

    Returns:
        A Hugging Face Dataset object loaded from disk.
    """


    output_dir = data_path("falcon-refinedweb")

    # Avoid multiprocessing issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"Streaming {num_samples} samples from Falcon-RefinedWeb...")
    
    # Stream the dataset
    stream_ds = load_dataset(
        "tiiuae/falcon-refinedweb",
        split="train",
        streaming=True,
    )
    
    # Shuffle and take samples
    stream_ds = stream_ds.shuffle(seed=seed, buffer_size=buffer_size)
    stream_ds = stream_ds.take(num_samples)
    
    # Collect samples into a list
    samples = []
    for sample in stream_ds:
        samples.append(sample)
    
    print(f"Collected {len(samples)} samples")
    
    # Convert to Dataset
    if samples:
        dataset = Dataset.from_dict({
            key: [sample[key] for sample in samples]
            for key in samples[0].keys()
        })
    else:
        raise ValueError("No samples collected!")
    
    # Save to disk
    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    print(f"Dataset saved! You can now load it with: load_from_disk('{output_dir}')")
    
    # Clean up any artifacts created in project root
    cleanup_hf_artifacts()
    
    return dataset

def download_mnli(): 
    """
    Download the MultiNLI dataset from the Hugging Face Hub.
    Returns:
        A Hugging Face Dataset object containing the MultiNLI data.
    """
    dataset = load_dataset("nyu-mll/multi_nli")
    return dataset



DOWNLOAD_DATASETS_REGISTRY = {
    # fine-tuning datasets
    "alpaca": download_alpaca,
    "mnli": download_mnli,

    # pretraining datasets
    "fineweb-edu": download_fineweb_edu,
    "falcon-refinedweb": download_falcon_refinedweb,
}

def download_bitlab_dataset(dataset_name: str):
    if dataset_name not in DOWNLOAD_DATASETS_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not found")
    return DOWNLOAD_DATASETS_REGISTRY[dataset_name]()