# Build the official base first from the Agenthon Track-1 public repository:
# docker build -t finance-bench-sandbox:latest -f docker/sandbox.Dockerfile .
ARG BASE_IMAGE=finance-bench-sandbox:latest
FROM ${BASE_IMAGE}

ARG AGENT_VERSION=0.8.0
ARG VCS_REF=unknown

WORKDIR /opt/qfa-agent
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
ENV PYTHONPATH=/opt/qfa-agent/src

# Public tasks sometimes describe the legacy /app/input and /app/data paths.
# Preserve those spellings while the official harness mounts the task at
# /input. /app/output remains a real mount point supplied by the harness.
RUN mkdir -p /app && ln -sfn /input /app/input && ln -sfn /input/environment/data /app/data

LABEL qfbench2.interface_version="2.0" \
      org.opencontainers.image.title="aptapt Agenthon T1 agent" \
      org.opencontainers.image.version="${AGENT_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/zolton-liu/aptapt"

# The harness supplies `solve` as the leading positional argument.  The CLI
# accepts that form and also works as the installed `solve` executable.
ENTRYPOINT ["python", "-m", "qfa_agent.cli"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
