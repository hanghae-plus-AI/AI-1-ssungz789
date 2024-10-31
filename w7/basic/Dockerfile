FROM python:3.10.15

RUN mkdir /llm_rag
WORKDIR /llm_rag
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x entrypoint.sh
EXPOSE 8501
CMD ["./entrypoint.sh"]
