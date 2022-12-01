FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev libomp-dev libopenblas-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /predict_app/requirements.txt

WORKDIR /predict_app

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . /predict_app

ENTRYPOINT [ "python3" ]

CMD [ "predict_app/server.py" ]
