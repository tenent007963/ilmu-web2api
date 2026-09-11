FROM python:3.11-alpine

WORKDIR /app

# Install the required requests module[cite: 1]
RUN pip install --no-cache-dir requests

# Copy the application script[cite: 1]
COPY ilmu_web2api.py .

EXPOSE 6769

ENTRYPOINT ["python", "ilmu_web2api.py"]