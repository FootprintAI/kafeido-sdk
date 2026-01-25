# Installation Guide

## Development Installation

To install the SDK in development mode:

```bash
cd /Users/hsinhoyeh/Workspaces/github/footprintai/kafeido-sdk
pip install -e .
```

This will install the package in "editable" mode, allowing you to make changes to the code and test them immediately.

## Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes testing and linting tools:
- pytest
- pytest-asyncio
- pytest-cov
- mypy
- ruff
- black
- respx

## Install with Async Support

```bash
pip install -e ".[async]"
```

This adds HTTP/2 support for better async performance.

## Verification

After installation, verify it works:

```python
python -c "from kafeido import OpenAI; print('✓ Kafeido SDK installed successfully')"
```

## Running Tests

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=kafeido --cov-report=term-missing

# Run type checking
mypy kafeido/

# Run linting
ruff check kafeido/
```

## Building for Distribution

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# - dist/kafeido-0.1.0.tar.gz (source distribution)
# - dist/kafeido-0.1.0-py3-none-any.whl (wheel)
```

## Publishing to PyPI

```bash
# Test PyPI (recommended first)
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

## Quick Test

```bash
# Set your API key
export KAFEIDO_API_KEY="sk-..."

# Run an example
python examples/chat_completion.py

# List models
python examples/list_models.py
```
