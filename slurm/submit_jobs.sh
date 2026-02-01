#!/bin/bash
# Usage: bash submit_jobs.sh <yaml_dir> [--partition=<name>] [--workers=<n>]
#
# Defaults: partition=a100, workers=12

YAML_DIR="$1"
shift

PARTITION="a100"
WORKERS=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --partition=*) PARTITION="${1#*=}"; shift ;;
        --workers=*)   WORKERS="${1#*=}}";  shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Validation ---
if [[ -z "$YAML_DIR" ]]; then
    echo "Usage: bash submit_jobs.sh <yaml_dir> [--partition=<name>] [--workers=<n>]"
    exit 1
fi
if [[ ! -d "$YAML_DIR" ]]; then
    echo "Error: '$YAML_DIR' is not a directory."
    exit 1
fi

# Collect yamls (sorted for reproducible ordering)
mapfile -t YAML_FILES < <(find "$YAML_DIR" -maxdepth 1 \( -name "*.yaml" -o -name "*.yml" \) | sort)

if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
    echo "No .yaml/.yml files found in '$YAML_DIR'."
    exit 1
fi

# --- Submit ---
echo "Submitting ${#YAML_FILES[@]} job(s) | partition=$PARTITION | workers=$WORKERS"
echo "---------------------------------------------------------------"

for yaml in "${YAML_FILES[@]}"; do
    # Derive a clean job name from the filename (no extension)
    JOB_NAME=$(basename "$yaml" | sed 's/\.\(yaml\|yml\)$//')

    echo "  -> $yaml  (job: $JOB_NAME)"
    sbatch \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --cpus-per-task="$WORKERS" \
        --export=ALL,YAML_PATH="$(realpath "$yaml")",NUM_WORKERS="$WORKERS" \
        run_job.sh
done

echo "---------------------------------------------------------------"
echo "Done. Check with: squeue -u $(whoami)"