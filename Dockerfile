FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY . .

CMD streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0
