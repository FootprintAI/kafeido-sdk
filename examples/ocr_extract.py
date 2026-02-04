"""OCR extraction example using Kafeido SDK."""

import time
from kafeido import OpenAI

client = OpenAI()

# Sync OCR extraction
result = client.ocr.extractions.create(
    model_id="deepseek-ocr",
    file_id="file-123",
    mode="markdown",
)
print(f"Extracted text:\n{result.text}")

# With grounding mode (bounding boxes)
result = client.ocr.extractions.create(
    model_id="deepseek-ocr",
    file_id="file-123",
    mode="grounding",
)
print(f"\nText: {result.text}")
if result.regions:
    for region in result.regions:
        print(f"  Region: '{region.text}' at ({region.x1},{region.y1})-({region.x2},{region.y2})")

# Async OCR extraction
job = client.ocr.extractions.create_async(
    model_id="deepseek-ocr",
    storage_key="org_123/document.pdf",
)
print(f"\nAsync OCR job: {job.job_id}")

while True:
    result = client.ocr.extractions.get_result(job_id=job.job_id)
    if result.status == "completed":
        print(f"Result: {result.result.text[:100]}...")
        break
    elif result.status == "failed":
        print(f"Error: {result.error}")
        break
    time.sleep(2)
