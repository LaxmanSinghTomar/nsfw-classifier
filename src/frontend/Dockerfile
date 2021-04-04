FROM ubuntu:20.04

WORKDIR /nsfw-classifier

COPY . nsfw-classifier/

RUN apt update \
    && apt install -y software-properties-common \ 
    && apt-get update \ 
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.7

RUN apt-get update\
    && apt-get install -y python3-pip \ 
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN /bin/bash -c "cd nsfw-classifier \ 
    && pip3 install -r requirements.txt"
    
EXPOSE 8501

CMD /bin/bash -c "streamlit run src/frontend/app.py"



