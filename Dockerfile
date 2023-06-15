FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install -y python3-pip

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["python", "src\main.py"]
