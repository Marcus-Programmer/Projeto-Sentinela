
# 🛡️ Projeto Sentinela

## 🎯 O Problema (A "Dor")

A internet moderna é construída sobre conteúdo gerado pelo usuário (UGC). No entanto, o volume massivo de publicações diárias tornou a moderação manual de conteúdo uma tarefa hercúlea, cara e, acima de tudo, psicologicamente desgastante para os moderadores humanos. Plataformas online enfrentam um dilema constante: como proteger suas comunidades contra discursos de ódio, assédio e abuso sem criar um sistema de vigilância insustentável?

A falha em moderar conteúdo de forma eficaz não apenas prejudica a saúde mental dos usuários e a segurança da comunidade, mas também expõe as plataformas a riscos legais e de reputação. A necessidade de uma solução escalável, rápida e inteligente nunca foi tão crítica.

## 📊 Justificativa e Relevância (Dados Reais)

A urgência deste problema é confirmada por dados alarmantes sobre o cenário online, especialmente no Brasil:

- **Crescimento do Discurso de Ódio:** Em 2022, as denúncias de crimes envolvendo discurso de ódio na internet no Brasil cresceram 67,7% em comparação com 2021, totalizando mais de 74.000 queixas, segundo a SaferNet Brasil.
- **Vulnerabilidade de Crianças e Adolescentes:** Uma pesquisa da TIC Kids Online Brasil de 2024 revelou que 3 em cada 10 crianças e adolescentes (29%) entre 9 e 17 anos já enfrentaram situações ofensivas ou discriminatórias na internet.
- **O Fardo da Moderação Humana:** Estudos internacionais apontam que mais de 50% dos moderadores de conteúdo relatam sintomas de estresse pós-traumático (PTSD) devido à exposição constante a material gráfico e violento.

## 💡 Nossa Solução

Este projeto aborda o problema através de um sistema distribuído composto por três microerviços principais:

- **API Gateway:** Ponto de entrada central que recebe todo o conteúdo a ser analisado e orquestra a comunicação entre os outros agentes.
- **Agente Analisador:** Um serviço de IA que utiliza um modelo de PLN para classificar o texto como "abuso" ou "normal".
- **Agente Encaminhador:** Um segundo agente de IA que utiliza um modelo de NER para identificar e anonimizar dados sensíveis no texto.

## 🚀 Como Executar o Projeto

### Passo 1: Pré-requisitos

Certifique-se de que você tem os seguintes softwares instalados:

- Docker e Docker Compose
- Python 3.8+ e pip (para o script de preparação)

### Passo 2: Preparação do Ambiente (Passo Único e Essencial)

Para que o projeto funcione offline e em redes restritivas, o modelo de IA do agente-analisador precisa ser baixado previamente.

Instale as bibliotecas Python necessárias (no seu computador local):

```bash
pip install transformers torch
```

Execute o script de download do modelo:  
Navegue até a pasta `agente-analisador` e execute o script. Isso irá baixar o modelo e salvá-lo na pasta `agente-analisador/local_sentiment_model`.

```bash
# Navegue para a pasta
cd agente-analisador

# Execute o script
python save_model.py
```

> ⚠️ **Atenção:** Este passo deve ser executado apenas uma vez, em uma rede sem restrições de internet.

### Passo 3: Construção e Execução dos Contêineres

Após a preparação do ambiente, volte para a pasta raiz do projeto e use o Docker Compose.

- **Construa as imagens Docker:**  
Este comando irá criar as imagens para cada serviço, copiando o modelo já baixado para dentro da imagem do agente-analisador.

```bash
docker-compose build
```

- **Inicie os serviços:**

```bash
docker-compose up
```

### Passo 4: Testando o Sistema

Com os contêineres em execução, você pode testar o endpoint principal.

- Para testar no **Windows (CMD/PowerShell):**

```bash
curl -X POST "http://localhost:8080/analisar-e-encaminhar" -H "Content-Type: application/json" -d "{\"texto\": \"O João me ameaçou em Belo Horizonte e disse que vai me machucar.\"}"
```

- Para testar no **Linux ou macOS:**

```bash
curl -X POST "http://localhost:8080/analisar-e-encaminhar" -H "Content-Type: application/json" -d '{"texto": "O João me ameaçou em Belo Horizonte e disse que vai me machucar."}'
```

Uma resposta de sucesso indicará que o texto foi classificado como "abuso" e encaminhado.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.9  
- **Frameworks de API:** FastAPI  
- **Containerização e Orquestração:** Docker e Docker Compose  
- **Inteligência Artificial:**
  - Hugging Face Transformers
  - spaCy (para NER)  
- **Comunicação:** API REST (HTTP)

## 📚 Referências

- Agência Brasil (2023): *Denúncias de crimes com discurso de ódio na internet crescem em 2022*.  
- Agência Brasil (2024): *Três em cada dez crianças e adolescentes foram ofendidos na internet*.  
- Fusion CX (2024): *Content Moderation Challenges*.

## 👥 Membros do Grupo

- [Nome do Aluno 1]  
- [Nome do Aluno 2]  
- [Nome do Aluno 3]  
- [Nome do Aluno 4]
