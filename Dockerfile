FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  libgomp1 \
  python3-pip \
  awscli \
  zip \
  ffmpeg \
  unzip

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY . /home/app
WORKDIR /home/app

RUN uv sync
ENV PATH=".venv/bin:$PATH"

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
