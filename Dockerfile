ARG BASE
FROM ${BASE}

WORKDIR /src

ENV PYTHONPATH=/src
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV AWS_ACCESS_KEY_ID=GOOG1ESPDYZTDM2J5ILHELSN7R4F75HZCL5XH6YEWJ2ZWZEOUPX7ISZW2QCHA
ENV AWS_SECRET_ACCESS_KEY=Vi4E8zfx0lXnVVwiQhEeUMZSsW7jg2eb6mH38e5C
ENV AWS_ENDPOINT=https://storage.googleapis.com
ENV AWS_USE_SSL=1

COPY bots ./bots
COPY common ./common
COPY db ./db
COPY grafana ./grafana
COPY jobs ./jobs
COPY models ./models
COPY preprocessing ./preprocessing
COPY tests ./tests
COPY tools ./tools
COPY workers ./workers

ENTRYPOINT ["python"]
