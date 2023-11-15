FROM python:3.8-slim
WORKDIR /usr/src/app
COPY sparse_recommender.py .
COPY test_sparse_recommender.py .
CMD [ "python", "./sparse_recommender.py" ]