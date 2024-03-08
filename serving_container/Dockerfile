FROM python:3.8
WORKDIR /usr/src/app

COPY /serving_container ./serving_container
RUN pip install --no-cache-dir -r serving_container/requirements.txt

ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict

ENTRYPOINT ["python", "serving_container/serve_model.py"]

ENV PYTHONPATH /usr/src/app