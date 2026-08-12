FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml requirements.txt ./
RUN uv pip install --system -r requirements.txt

COPY src/ ./src/
COPY demo.py .
COPY .env.example .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
