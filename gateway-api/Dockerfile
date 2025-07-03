# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo de dependências
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o resto do código da aplicação
COPY . .

# Comando para iniciar o servidor da API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
