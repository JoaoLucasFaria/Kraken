# Kraken

**Trabalho da disciplina de Inteligência Artificial (2025/2) - UFSJ**

**Alunos:** Arthur Henrique Moreira Santos, João Lucas de Vilas Bôas Faria, Leonardo Sousa Bahia e João Vitor Simão

---

## Requisitos

- Python 3.8+
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `peft`, `llama-cpp-python`, `ipywidgets`
- Instale as dependências com: `pip install -r requirements.txt`

---

## Modelos Necessários

Este projeto utiliza o modelo **Llama 3.1 8B Instruct** em duas variações:

### 1. Modelo GGUF
- **Download:** [Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF)
- **Variante recomendada:** `Meta-Llama-3.1-8B-Instruct.Q2_K.gguf`
- **Localização no projeto:** `src/models/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf`
- **Como baixar:**
  ```bash
  hf download meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct.Q2_K.gguf" \
    --local-dir src/models
  ```

### 2. Modelo Fine-tuned
- O modelo fine-tuned foi treinado em Google Colab devido às exigências computacionais
- Localização no projeto: `src/models/llama_finetuned/`
- Se não disponível, o sistema utilizará automaticamente o modelo GGUF como alternativa

---

## Estrutura do Projeto

```
.
├── code.ipynb                          # Notebook principal
├── input.csv                           # Dados de entrada (monstros D&D)
├── outputs/
│   ├── monsters_clean.csv              # Dados processados
│   ├── monsters_train.csv              # Features para clustering
│   └── descricao_encontro.txt          # Saída final gerada
├── src/
│   ├── model_loader.py                 # Carregador de modelos GGUF
│   ├── finetuned_model_loader.py       # Carregador de modelos fine-tuned
│   ├── scrapper.py                     # Web scraper para descrições
│   └── models/
│       ├── Meta-Llama-3.1-8B-Instruct-Q2_K.gguf
│       └── llama_finetuned/
└── finetune_colab/
    └── finetune.ipynb                  # Notebook para fine-tuning em Colab
```

---

## Como Executar

### Passo 1: Preparação
1. Clone ou baixe o repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Baixe o modelo GGUF (veja seção "Modelos Necessários")

### Passo 2: Processamento de Dados
Execute as seguintes células do `code.ipynb` sequencialmente:
- **ETAPA 0:** Leitura do CSV com validação de encoding
- **ETAPA 1:** Seleção de colunas essenciais
- **ETAPA 2:** Limpeza e transformação dos dados
- **ETAPA 3:** Transformação com one-hot encoding
- **Exportação final:** Gera `monsters_clean.csv` e `monsters_train.csv`

### Passo 3: Clustering Temático
Execute a célula **Etapa 4** para aplicar K-Means:
- Agrupa monstros em 4 clusters baseado em tipo, ambiente e tamanho
- Normaliza os dados para melhor distribuição

### Passo 4: Cálculo de Dificuldade
Execute a célula **Etapa 5** para ativar as funções de:
- Cálculo do valor do grupo (baseado nos níveis dos jogadores)
- Avaliação da dificuldade do encontro

### Passo 5: Geração de Encontros
Execute as células **Geração de Encontros** e **Testes**:
1. Preencha os parâmetros desejados:
   - Níveis dos jogadores
   - Dificuldade (fácil, médio, difícil, impossível)
   - Tipo de inimigos (qualquer tipo ou específico)
   - Ambiente (qualquer ambiente ou específico)
   - Tamanho dos inimigos (qualquer tamanho ou específico)
   - Quantidade de oponentes

2. O sistema testará até 15 tentativas para encontrar o melhor encontro que se aproxime da dificuldade solicitada

### Passo 6: Interface Interativa
Execute a célula **Interface do Usuário**:
- Uma interface interativa será exibida
- Preencha os parâmetros do encontro
- Clique em "Gerar combate" para obter a sugestão

### Passo 7: Geração de Descrição (Llama)
Execute a célula **Descrição de Cenas**:
- Carrega automaticamente o modelo fine-tuned (se disponível) ou o GGUF
- Envia os dados do encontro ao modelo de IA
- Gera uma descrição atmosférica do encontro em português
- Salva a descrição em `outputs/descricao_encontro.txt`

---

## Fluxo de Execução Recomendado

```
1. Instalar dependências
   ↓
2. Baixar modelo GGUF
   ↓
3. Executar ETAPA 0-4 (uma única vez)
   ↓
4. Executar ETAPA 5 (uma única vez)
   ↓
5. Executar "Carregamento de modelo" (uma única vez)
   ↓
6. Executar "Interface do Usuário" (uma única vez)
   ↓
7. Preencher formulário e gerar encontro
   ↓
8. Executar "Descrição de Cenas" para gerar descrição final
```

---

## Saída do Programa

O programa gera um arquivo de texto em `outputs/descricao_encontro.txt` contendo:
- Descrição atmosférica do encontro
- Detalhes do ambiente
- Comportamento e aparência dos monstros
- Tom narrativo adequado (sombrio, épico, etc.)

---

## Notas Técnicas

- **GPU recomendada:** NVIDIA CUDA para melhor desempenho
- **CPU:** Suportado, mas mais lento
- **Tempo de execução:** Varia dependendo do hardware
- **Memória necessária:** Mínimo 8GB RAM (16GB recomendado)

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| Modelo GGUF não encontrado | Verifique se está em `src/models/` |
| Erro ao carregar fine-tuned | Sistema usará GGUF automaticamente como fallback |
| Memória insuficiente | Reduza `n_gpu_layers` em `model_loader.py` |



