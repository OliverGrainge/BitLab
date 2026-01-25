import argparse
import sys
from typing import Iterable, Sequence

from src.data.download import download_bitlab_dataset, DOWNLOAD_DATASETS_REGISTRY
from src.models.download import download_bitlab_model, DOWNLOAD_MODELS_REGISTRY


def available_datasets() -> Sequence[str]:
    """Return dataset names in a stable order for UX."""
    return sorted(DOWNLOAD_DATASETS_REGISTRY.keys())


def available_models() -> Sequence[str]:
    """Return model names in a stable order for UX."""
    return sorted(DOWNLOAD_MODELS_REGISTRY.keys())


def download_datasets(ds_names: Iterable[str]) -> None:    
    for name in ds_names:
        download_bitlab_dataset(name)
    print("[done] datasets")


def download_models(model_names: Iterable[str]) -> None:
    """Download one or more models by name."""
    for name in model_names:
        download_bitlab_model(name)
    print("[done] models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download datasets and models from their respective registries."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help=f"One or more dataset names. Available: {', '.join(available_datasets())}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help=f"One or more model names. Available: {', '.join(available_models())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all registered datasets and models.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and models and exit.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        datasets = available_datasets()
        models = available_models()
        if datasets:
            print("Datasets:")
            print("\n".join(f"  {ds}" for ds in datasets))
        if models:
            print("Models:")
            print("\n".join(f"  {m}" for m in models))
        if not datasets and not models:
            print("No datasets or models available.")
        return 0

    if not args.datasets and not args.models and not args.all:
        parser.error("one of the arguments --datasets --models --all is required")

    # Determine what to download
    download_datasets_list = []
    download_models_list = []

    if args.all:
        download_datasets_list = available_datasets()
        download_models_list = available_models()
    else:
        if args.datasets:
            download_datasets_list = args.datasets
        if args.models:
            download_models_list = args.models

    # Validate datasets
    if download_datasets_list:
        unknown = [n for n in download_datasets_list if n not in DOWNLOAD_DATASETS_REGISTRY]
        if unknown:
            print(
                f"Error: unknown dataset(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(available_datasets())}",
                file=sys.stderr,
            )
            return 2

    # Validate models
    if download_models_list:
        unknown = [n for n in download_models_list if n not in DOWNLOAD_MODELS_REGISTRY]
        if unknown:
            print(
                f"Error: unknown model(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(available_models())}",
                file=sys.stderr,
            )
            return 2

    # Download datasets and models
    try:
        if download_datasets_list:
            download_datasets(download_datasets_list)
        if download_models_list:
            download_models(download_models_list)
        if not download_datasets_list and not download_models_list:
            print("Nothing to download.")
    except Exception as e:
        print(f"Error while downloading: {e}", file=sys.stderr)
        return 1

    # Force cleanup before exit to avoid segfault
    import gc
    gc.collect()
    
    return 0


if __name__ == "__main__":
    import os
    exit_code = main()
    # Use os._exit to bypass Python cleanup that causes segfault with datasets library
    os._exit(exit_code)