FROM python:3.12-slim@sha256:e70237b264442328526ff29d6b3c43bd8abcf6cbab78a949322fb69b7d91d6b6
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 h5py==3.11.0 sympy==1.13.3 matplotlib==3.9.2 pandas==2.2.3
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib PYTHONHASHSEED=0
COPY scicode /opt/scicode
ENV PYTHONPATH=/opt
WORKDIR /work
USER 65534:65534
