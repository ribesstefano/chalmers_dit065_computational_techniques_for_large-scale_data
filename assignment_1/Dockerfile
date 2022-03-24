FROM python:3.7-slim-buster

RUN mkdir -p /usr/src/app && useradd -ms /bin/bash user && chown -R user:user /usr/src/app
USER user
WORKDIR /usr/src/app
COPY --chown=user:user requirements.txt .
RUN python -m pip install --user -r requirements.txt
