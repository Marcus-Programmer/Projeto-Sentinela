import requests
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextoParaAnalise(BaseModel):
    texto: str

@app.post("/analisar-e-encaminhar")
async def analisar_e_encaminhar(data: TextoParaAnalise):
    """
    Ponto de entrada do sistema. Recebe um texto, o envia para análise
    e, se necessário, o encaminha para registro e anonimização.
    """
    texto_original = data.texto
    logger.info(f"Gateway: Novo texto recebido para orquestração.")

    try:
        # 1. Chamar o Agente Analisador
        logger.info("Gateway: Comunicando com o Agente Analisador...")
        url_analisador = "http://analisador:8000/analise"
        response_analisador = requests.post(url_analisador, json={"texto": texto_original})
        response_analisador.raise_for_status()  # Lança exceção para erros HTTP (4xx ou 5xx)
        
        resultado_analise = response_analisador.json()
        classificacao_ia = resultado_analise.get("classificacao")
        logger.info(f"Gateway: Análise recebida com a classificação '{classificacao_ia}'.")

        # 2. Se for abuso, chamar o Agente Encaminhador
        if classificacao_ia == "abuso":
            logger.info("Gateway: Comunicando com o Agente Encaminhador...")
            url_encaminhador = "http://encaminhador:8001/registrar-denuncia"
            
            payload_denuncia = {
                "texto": texto_original,
                "classificacao": classificacao_ia,
                "confianca_modelo": resultado_analise.get("confianca_do_modelo")
            }
            
            response_encaminhador = requests.post(url_encaminhador, json=payload_denuncia)
            response_encaminhador.raise_for_status()
            
            logger.info("Gateway: Denúncia registrada e anonimizada com sucesso.")
            return {"status": "Denúncia encaminhada com sucesso", "classificacao_ia": classificacao_ia}
        
        else:
            logger.info("Gateway: Texto considerado normal. Processo finalizado.")
            return {"status": "Análise concluída, texto considerado normal.", "classificacao_ia": classificacao_ia}

    except requests.exceptions.RequestException as e:
        logger.error(f"Gateway: Erro de comunicação com um serviço interno: {e}")
        raise HTTPException(status_code=503, detail=f"Serviço interno indisponível: {e}")
    except Exception as e:
        logger.error(f"Gateway: Ocorreu um erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro inesperado no gateway.")

@app.get("/")
def health_check():
    """
    Endpoint de verificação de saúde para o Docker.
    """
    return {"status": "Gateway API está online"}
