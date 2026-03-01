

---
![wave-gif](https://camo.githubusercontent.com/89a46b75cb2af1de643c4ae5e510aff5c0fa30e7e2a9cdfa5e4ab46eae39a19e/68747470733a2f2f692e696d6775722e636f6d2f315a76566b44632e676966)


# Привет, меня зовут Сергей!

## Я AI/ML Engineer с более чем 3-летним опытом разработки end-to-end NLP, RAG, AI ассистентов и агентов!

> AI/ML-инженер, специализирующийся на построении интеллектуальных систем — от RAG-пайплайнов до production-готовых AI-агентов. Помогаю компаниям автоматизировать процессы с помощью LLM.

### В своей работе я:

⭑ Создаю автономные AI-агенты на базе LangGraph/CrewAI для решения бизнес-задач и интеграции с внутренними сервисами

⭑ Использую контекстную инженерию (LLMChainExtractor/xprovence-reranker-bgem3-v1) для уменьшения cost  и latency 

⭑ Развертываю и оптимизирую векторные и графовые БД (Chroma, FAISS, LightRAG)

⭑ Реализую автоматизацию пайплайнов обработки и генерации через LangChain и FastAPI

⭑ Использую комбинированный поиск в EnsembleRetriever с косинусным сходством и поиском по ключевым словам

⭑ Использую в своей работе Crawl4AI для автоматизированного парсинга

⭑ Дообучаю и оптимизирую эмбеддинги на кастомных датасетах используя LoRA и PEFT

⭑ Интегрирую LLM (GPT-4o-mini, Hugging Face Transformers) в production-среду

⭑ Обеспечиваю мониторинг и масштабируемость AI-сервисов (LangSmith/LangFuse)

## 🚀 Мои проекты

| Проект | Описание | Стек |
|--------|----------|------|
| [🚗 LangGraph DIY Car Agent](https://github.com/SergeyKarpenko1/LangGraph-DIY-car-agent) | Многошаговый AI-ассистент для DIY детейлинга автомобиля на LangGraph. Гибридный ретривер (ChromaDB MMR + BM25 + reranker) вынесен в отдельный MCP-сервер. Включает human-in-the-loop через `interrupt`, SSE backend на FastAPI и Streamlit UI с отображением внутренних шагов графа. Поддерживает Docker/docker-compose деплой. | LangGraph, LangChain, FastAPI, ChromaDB, BM25, Streamlit, Docker |
| [🤖 SGR Agent ING](https://github.com/SergeyKarpenko1/SGR_Agent_ING) | Консольный tool-calling агент с планированием и верификацией ответов на основе Википедии. Пайплайн: `GeneratePlanTool` → `wiki_search` / `wiki_page_extract` → `FinalAnswerTool`. Реализован `SafeToolCallingAgent` для graceful fallback при нестандартных ответах моделей. Поддерживает **A2A (Agent-to-Agent) протокол** — агент публикует `AgentCard` и принимает запросы от других агентов по HTTP. | Python, OpenRouter, LangChain, A2A, FastAPI, uv |
| [⚡ FastAPI + llama.cpp](https://github.com/SergeyKarpenko1/FastAPI_with_Llamacpp) | Production-ready REST API сервис для запуска локальных LLM (GGUF-формат) через `llama-cpp-python`. Включает эндпоинты `/health` и `/predict`, Pydantic-схемы, асинхронный load generator с параллельными воркерами для нагрузочного тестирования. Полная контейнеризация через Docker и docker-compose (API + load generator). | FastAPI, llama.cpp, Uvicorn, Docker, httpx, Pydantic |
| [📚 Moodle RAG](https://github.com/SergeyKarpenko1/Moodle_RAG) | End-to-end RAG-система для образовательной платформы Moodle. Полный пайплайн: краулинг документации через `crawl4ai` + Playwright → очистка и нарезка на чанки → индексация в ChromaDB → RAG API на FastAPI с генерацией ответов через `mlx-lm` (Qwen2.5-7B-Instruct-4bit). Возвращает ответ со ссылками на источники и YouTube-материалы. | crawl4ai, LangChain, ChromaDB, FastAPI, mlx-lm, Qwen2.5 |
| [📝 CrewAI README Generator](https://github.com/SergeyKarpenko1/CrewAI_creating_readmemd) | Мультиагентная система автоматической генерации README.md на базе CrewAI. Три специализированных агента (сборщик структуры → ридер контента → генератор документации) работают последовательно через кастомный инструмент `RecursiveDirectoryScanner`. Поддерживает двуязычный вывод (EN/RU), конфигурацию через YAML. Интегрирован с Claude 3.7 Sonnet via OpenRouter. | CrewAI, LangChain, Claude 3.7 Sonnet, OpenRouter, Python |
| [⚖️ RAG Legal Assistant](https://github.com/SergeyKarpenko1/RAG_Legal_assistant) | RAG-прототип для анализа судебных решений и юридических документов РФ. Полный ML-пайплайн: парсинг PDF через Crawl4AI, **файн-тюнинг эмбеддингов** на домен-специфичных юридических данных (LoRA + SentenceTransformers), сравнительный бенчмарк стратегий чанкинга, векторный поиск через ChromaDB. Поддерживает решения арбитражных судов и судов общей юрисдикции. | ChromaDB, SentenceTransformers, LoRA, Crawl4AI, LangChain, Jupyter |

### Стек технологий:

#### **Обработка данных**:

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org) 
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23EE4C2C.svg?style=for-the-badge&logo=Matplotlib&logoColor=white)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-%23007ACC.svg?style=for-the-badge&logo=Seaborn&logoColor=white)](https://seaborn.pydata.org/) 
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%23336791.svg?style=for-the-badge&logo=PostgreSQL&logoColor=white)](https://www.postgresql.org/)
[![Spark](https://img.shields.io/badge/Spark-%23E25A1C.svg?style=for-the-badge&logo=Apache%20Spark&logoColor=white)](https://spark.apache.org/)

#### **Машинное обучение**:

[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%230078D7.svg?style=for-the-badge&logo=XGBoost&logoColor=white)](https://xgboost.readthedocs.io/en/latest/)
[![CatBoost](https://img.shields.io/badge/CatBoost-%23EE4C2C.svg?style=for-the-badge&logo=CatBoost&logoColor=white)](https://catboost.ai/)


#### **Глубокое обучение**

[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-%230083BD?style=for-the-badge&logo=python&logoColor=white)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/Transformers-%23FF4500?style=for-the-badge&logo=transformers&logoColor=white)](https://huggingface.co/transformers/) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD700?style=for-the-badge&logo=Hugging%20Face&logoColor=black)](https://huggingface.co/) [![OpenAI](https://img.shields.io/badge/OpenAI-%2301051E?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/) [![LoRA](https://img.shields.io/badge/LoRA-%23FFA500?style=for-the-badge&logo=LoRA&logoColor=white)](https://arxiv.org/abs/2106.09685) [![PEFT](https://img.shields.io/badge/PEFT-%23007ACC?style=for-the-badge&logo=peft&logoColor=white)](https://github.com/huggingface/peft) [![RAG](https://img.shields.io/badge/RAG-%23FFD700?style=for-the-badge&logo=Hugging%20Face&logoColor=black)](https://huggingface.co/blog/rag) [![LangChain](https://img.shields.io/badge/LangChain-%23007ACC?style=for-the-badge&logo=LangChain&logoColor=white)](https://langchain.com/) [![Chroma](https://img.shields.io/badge/Chroma-%2300C4CC?style=for-the-badge&logo=chroma&logoColor=white)](https://www.trychroma.com/) [![Crawl4AI](https://img.shields.io/badge/Crawl4AI-%23000000?style=for-the-badge)](#) [![CrewAI](https://img.shields.io/badge/CrewAI-%23007ACC?style=for-the-badge)](#)


#### **DevOps и прочее**:

[![Git](https://img.shields.io/badge/Git-%23F05032.svg?style=for-the-badge&logo=Git&logoColor=white)](https://git-scm.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com/)
[![Aiogram](https://img.shields.io/badge/Aiogram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://docs.aiogram.dev/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)


Всегда рад новым возможностям для сотрудничества:

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/Karpenko_Sergey1)

</div>

---
