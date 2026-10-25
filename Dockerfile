# n2-sqd: N2 SQD hackathon environment (Qiskit Fall Fest 2026 Warsaw, Track 4)
# Base pinned by digest (python 3.13.17, resolved 2026-10-23).
FROM python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39

ENV PYTHONPATH=/work \
    OMP_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    MKL_NUM_THREADS=16
WORKDIR /work
COPY requirements.txt requirements.lock /tmp/
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /tmp/requirements.txt -c /tmp/requirements.lock \
 && python -c "import pyscf, ffsim, qiskit, qiskit_ibm_runtime, qiskit_addon_sqd, qfest_grader; print('imports ok')"

CMD ["python"]
