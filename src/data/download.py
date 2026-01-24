from datasets import load_dataset, Dataset, load_from_disk
from src.data import dataset 


def download_alpaca(): 
    """
    Download the Alpaca dataset from the Hugging Face Hub.
    Returns:
        A Hugging Face Dataset object containing the Alpaca data.
    """
    dataset = load_dataset("tatsu-lab/alpaca")
    return dataset


def download_fineweb_edu(
    output_dir: str = "data/fineweb-edu",
    num_samples: int = 1000000,
    buffer_size: int = 10000,
    seed: int = 0,
):
    """
    Download and save a small subset of FineWeb-Edu to disk.

    Args:
        output_dir: directory to save the dataset
        num_samples: number of documents to download
        buffer_size: shuffle buffer for streaming
        seed: RNG seed

    Returns:
        A Hugging Face Dataset object loaded from disk.
    """
    from datasets import load_dataset, Dataset
    import os
    
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
    
    return dataset



DOWNLOAD_DATASETS_REGISTRY = {
    "alpaca": download_alpaca,
    "fineweb-edu": download_fineweb_edu,
}