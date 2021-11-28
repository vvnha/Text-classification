FROM python:3.6.13

# COPY bashrc /root/.bashrc
# WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
ADD api.py /api.py
ADD okteto-stack.yaml /okteto-stack.yaml
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./models models
COPY api.py api.py
COPY model.py model.py
COPY stopwords.txt stopwords.txt
COPY sw.py sw.py
COPY test.txt test.txt
COPY train.py train.py
COPY xuli.py xuli.py

EXPOSE 6868
CMD ["python3", "api.py"]