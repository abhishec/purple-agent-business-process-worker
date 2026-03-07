FROM public.ecr.aws/docker/library/python:3.12-slim

# Create non-root user (AgentBeats best practice)
RUN useradd -m -u 1000 agentbeats

WORKDIR /app

# Install system utilities (curl for health check, git for pip git+https deps)
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Fix: ensure /app is owned by agentbeats so RL/knowledge files can be written
# (case_log.json, knowledge_base.json, entity_memory.json, etc. all write here)
RUN chown agentbeats:agentbeats /app

COPY --chown=agentbeats:agentbeats requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=agentbeats:agentbeats src/ ./src/
COPY --chown=agentbeats:agentbeats main.py .

USER agentbeats

EXPOSE 9010

# AgentBeats-compatible entrypoint: accepts --host, --port, --card-url
ENTRYPOINT ["python", "main.py"]
CMD ["--host", "0.0.0.0", "--port", "9010"]
