FROM python:3.11-slim

WORKDIR /app

# Install the SDK
COPY pyproject.toml README.md LICENSE ./
COPY kafeido/ kafeido/
COPY examples/ examples/

RUN pip install --no-cache-dir .

# Default to an interactive Python shell with kafeido pre-imported
CMD ["python3", "-i", "-c", "from kafeido import OpenAI, AsyncOpenAI; print('Kafeido SDK v1.4.0 loaded.\\nUsage: client = OpenAI(api_key=\"sk-...\")\\nSee examples/ for usage patterns.')"]
