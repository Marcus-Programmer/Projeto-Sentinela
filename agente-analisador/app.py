import logging
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Carregando modelo de IA (Hugging Face) de um diretório local...")
classificador_sentimento = None
try:

    model_path = "/app/local_sentiment_model"
    
 
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
  
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    

    classificador_sentimento = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer
    )
    logger.info("Modelo de Análise de Sentimento carregado com sucesso do diretório local!")
except Exception as e:
    logger.error(f"FALHA CRÍTICA ao carregar modelo local: {e}", exc_info=True)

app = FastAPI()

class TextoParaAnalise(BaseModel):
    texto: str

@app.post("/analise")
async def analisar_texto(data: TextoParaAnalise) -> Dict:
    """
    Recebe um texto e o classifica como 'abuso' ou 'normal' usando o modelo de IA local.
    """
    if not classificador_sentimento:
        logger.error("Tentativa de análise, mas o modelo não está disponível.")
        raise HTTPException(status_code=503, detail="Serviço de IA de Análise indisponível.")

    logger.info("Classificando texto recebido...")
    
    try:
        resultado_analise = classificador_sentimento(data.texto)
        primeiro_resultado = resultado_analise[0]
        label = primeiro_resultado['label']
        score = primeiro_resultado['score']

        if label == "negative":
            classificacao = "abuso"
        else:
            classificacao = "normal"
            
        logger.info(f"Texto classificado como '{classificacao}' com confiança de {score:.2f}.")

        return {
            "classificacao": classificacao,
            "sentimento_detectado": label,
            "confianca_do_modelo": score
        }
    except Exception as e:
        logger.error(f"Erro inesperado durante a análise do texto: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno no processo de análise.")


@app.get("/")
def health_check():
    """
    Endpoint de verificação de saúde para o Docker.
    """
    return {"status": "Agente analisador está online"}