# Same base tag as the measured deployment. Pin its platform digest for your release.
FROM nvcr.io/nvidia/pytorch:26.07-py3
ARG TF_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46
RUN git clone https://github.com/bertholomus/TensorFold /opt/tensorfold && \
    git -C /opt/tensorfold checkout "${TF_COMMIT}"
COPY . /opt/spark1017
RUN python3 /opt/spark1017/tools/prepare_source.py /opt/tensorfold && \
    pip install --no-cache-dir -e '/opt/tensorfold[vision]' -e /opt/spark1017
ENV MAX_JOBS=1 TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    CUDA_CACHE_PATH=/cache/nv TRITON_CACHE_DIR=/cache/triton
ENTRYPOINT ["tensorfold"]
