from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define o nome do modelo e onde salvá-lo
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
save_directory = "./local_sentiment_model"

print(f"Baixando e salvando o modelo '{model_name}' para '{save_directory}'...")

# Baixa o 'tokenizer' e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Salva os arquivos na pasta local
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Download e salvamento concluídos com sucesso!")
print(f"A pasta '{save_directory}' foi criada.")