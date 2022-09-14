FROM python:3.10

ENV PORT 8501
ENV HOST 0.0.0.0

EXPOSE 8501

RUN apt-get update -y && apt-get install -y apt-transport-https
RUN apt-get install -y python3-pip && \
    apt-get install -y sudo curl git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install git-lfs=2.11.0 && \
    mkdir -p /app

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt && \
    git lfs pull -I word_embeddings2.pickle

COPY . /app


ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
