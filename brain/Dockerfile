FROM python:3.11-slim

# install dependencies
RUN pip install fastapi uvicorn[standard]

# copy app
WORKDIR /app
COPY . .

# default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
