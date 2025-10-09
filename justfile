# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config.yaml

# Install the repository
install:
  uv sync --extra cu126

# Run linting and formatting
lint: setup
  pre-commit run --all-files || pre-commit run --all-files

# Run ruff linting
ruff:
  uvx ruff check cosmos_transfer1/ scripts/ server/ --config ruff.toml
  uvx ruff format cosmos_transfer1/ scripts/ server/ --config ruff.toml

# Test environment setup
test-env:
  python scripts/test_environment.py

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions"

# Update the license
license: install
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md
  pre-commit run --files ATTRIBUTIONS.md || true

# Pre-release checks
release-check:
  just license
  pre-commit run --all-files --hook-stage manual

# Run the docker container
docker *args:
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it $(docker build -q .) {{args}}

# Build Docker image with tag
# docker-build tag="cosmos-transfer1:latest":
docker-build tag="nvcr.io/ebplj5vljyh4/dev/cosmos-transfer1:v1-uv-vaibhav":
  docker build -t {{tag}} .

# Download model checkpoints
download-checkpoints:
  python scripts/download_checkpoints.py

# Download diffusion example data
download-data:
  python scripts/download_diffusion_example_data.py

# Convert checkpoint from FSDP to TP format
convert-fsdp-to-tp input output:
  python scripts/convert_ckpt_fsdp_to_tp.py --input {{input}} --output {{output}}

# Convert checkpoint from TP to FSDP format
convert-tp-to-fsdp input output:
  python scripts/convert_ckpt_tp_to_fsdp.py --input {{input}} --output {{output}}

# Get T5 embeddings
get-t5-embeddings input output:
  python scripts/get_t5_embeddings.py --input {{input}} --output {{output}}

# Clean up cache files and build artifacts
clean:
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
  find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
  find . -name "*.pyc" -delete 2>/dev/null || true
  find . -name ".coverage" -delete 2>/dev/null || true
  find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Show available examples
examples:
  @echo "Available inference examples:"
  @ls examples/ | grep inference | sed 's/^/  - /'
  @echo ""
  @echo "Available training examples:"
  @ls examples/ | grep training | sed 's/^/  - /'