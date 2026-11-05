ARG BASE_IMAGE=python:3.12.13-slim-bookworm
ARG WORKER_BASE_IMAGE=${BASE_IMAGE}
ARG NODE_IMAGE=node:20-bookworm
FROM ${BASE_IMAGE} AS client-python
FROM ${NODE_IMAGE} AS client
COPY --from=client-python /usr/local/ /usr/local/
RUN ldconfig && useradd --create-home --uid 1001 agent && mkdir -p /workspace /run/gym-runtime && chown agent:agent /workspace
ENV PYTHONPATH=/opt/client PYTHONDONTWRITEBYTECODE=1
COPY gym/src/nexttransformer_gym/__init__.py gym/src/nexttransformer_gym/client.py gym/src/nexttransformer_gym/common.py /opt/client/nexttransformer_gym/
COPY gym/examples/make_candidate.py gym/tasks/architecture-research/environment/runtime_snapshot.py /opt/gym/
WORKDIR /workspace
USER agent

FROM ${WORKER_BASE_IMAGE} AS service
ARG INSTALL_WORKER_DEPS=1
ARG INSTALL_SANDBOX_DEPS=1
RUN if [ "$INSTALL_SANDBOX_DEPS" = "1" ]; then apt-get update && apt-get install -y --no-install-recommends bubblewrap && rm -r /var/lib/apt/lists/*; else bwrap --version; fi
RUN if [ "$INSTALL_WORKER_DEPS" = "1" ]; then pip install --no-cache-dir numpy==2.2.6 pyarrow==21.0.0 tiktoken==0.12.0 rustbpe==0.1.0 psutil==7.1.3 filelock==3.20.0 typing-extensions==4.15.0 setuptools==78.1.0 sympy==1.14.0 networkx==3.7 jinja2==3.1.6 fsspec==2026.9.0 && pip install --no-cache-dir --no-deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu; else python -c "import torch,numpy,pyarrow; assert torch.__version__.split('+')[0] == '2.9.1'"; fi
ENV PYTHONPATH=/opt/gym/src:/opt/platform PYTHONDONTWRITEBYTECODE=1 NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1
COPY nexttransformer /opt/platform/nexttransformer
COPY nanochat /opt/platform/nanochat
COPY results/runs /opt/platform/results/runs
COPY results/methods /opt/platform/results/methods
COPY configs /opt/platform/configs
COPY gym/src /opt/gym/src
COPY gym/examples/profiles.fixture.json /opt/gym/profiles.json
COPY gym/tasks/architecture-research/environment/entrypoint.sh /opt/gym/entrypoint.sh
ENTRYPOINT ["sh", "/opt/gym/entrypoint.sh"]

FROM ${BASE_IMAGE} AS verifier
ENV PYTHONPATH=/opt/gym/src PYTHONDONTWRITEBYTECODE=1
COPY gym/src /opt/gym/src
COPY gym/tasks/architecture-research/tests /tests
