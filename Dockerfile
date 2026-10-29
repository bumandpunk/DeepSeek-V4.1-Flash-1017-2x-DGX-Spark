# Official arm64 base. build.py also verifies any offline tag by image ID.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
FROM ${BASE_IMAGE}
ARG TF_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46
ARG SOURCE_SHA256
RUN git init /opt/tensorfold && \
    git -C /opt/tensorfold remote add origin https://github.com/bertholomus/TensorFold.git && \
    for attempt in 1 2 3; do \
      git -c http.version=HTTP/1.1 -C /opt/tensorfold fetch --depth 1 origin "${TF_COMMIT}" && break; \
      test "$attempt" -lt 3 || exit 1; sleep "$((attempt * 3))"; \
    done && \
    git -C /opt/tensorfold checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/tensorfold rev-parse HEAD)" = "${TF_COMMIT}"
COPY . /opt/spark1017
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python3 /opt/spark1017/tools/prepare_source.py /opt/tensorfold && \
    python3 /opt/spark1017/scripts/ngc_constraints.py /tmp/ngc-pins.txt && \
    pip install --no-cache-dir --no-build-isolation --report /opt/spark1017-pip-report.json -c /tmp/ngc-pins.txt \
      -c /opt/spark1017/docker/runtime-constraints.txt \
      -e '/opt/tensorfold[vision,grammar]' -e /opt/spark1017 && \
    python3 /opt/spark1017/scripts/image_manifest.py /opt/spark1017-image.json
ENV MAX_JOBS=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    CUDA_CACHE_PATH=/cache/nv TRITON_CACHE_DIR=/cache/triton
LABEL org.opencontainers.image.title="Spark1017" \
      org.opencontainers.image.version="0.2.0-alpha.1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.spark1017.source-sha256="${SOURCE_SHA256}" \
      org.spark1017.tensorfold-commit="${TF_COMMIT}"
WORKDIR /opt/spark1017
ENTRYPOINT ["tensorfold"]
