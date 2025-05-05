from pydantic import BaseModel, Field
from typing import Optional, List
from trustcall import create_extractor
from functools import lru_cache
from database.pg_vector import SupabaseVectorDB

from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

import configuration

# --------------------- LLM SETUP ---------------------
model = ChatVertexAI(model="gemini-2.0-flash-lite-001", temperature=0, max_tokens=200)


# --------------------- USER PROFILE ---------------------
class UserProfile(BaseModel):
    nome: Optional[str] = Field(None)
    sobrenome: Optional[str] = Field(None)
    email: Optional[str] = Field(None)
    telefone: Optional[str] = Field(None)
    necessidade: Optional[str] = Field(None)
    valor_desejado: Optional[str] = Field(None)
    urgencia: Optional[str] = Field(None)
    nivel_conhecimento_consorcio: Optional[str] = Field(None)
    disponibilidade_lance: Optional[str] = Field(None)
    finalidade: Optional[str] = Field(None)
    orcamento_mensal: Optional[str] = Field(None)
    tomada_decisao: Optional[str] = Field(None)


# --------------------- TRUSTCALL EXTRACTOR ---------------------
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile",
)

# --------------------- PROMPT SETUP ---------------------
# Agent instruction
MODEL_SYSTEM_MESSAGE = """
Você é um agente de atendimento e qualificação (SDR) especializado em consórcios. Seu papel é acolher leads de forma humanizada e conduzir uma conversa leve, 
estratégica e consultiva, com foco em entender o momento do cliente e responder suas dúvidas de forma prática e DIRETA.

Diretrizes:
1. NUNCA use emojis, linguagem informal ou gírias.
2. Nunca comece falando sobre consórcio. Dê espaço para o usuário guiar a conversa no início.
3. Não faça perguntas diretas, mas sim perguntas abertas que incentivem o lead a compartilhar informações.
4. Nunca faça mais de uma pergunta junto. Pergunte uma coisa de cada vez.
5. Sempre use as infomações técnicas relevantes sobre consórcios para responder às perguntas do usuário.

Se você tiver memória sobre este usuário, use essas informações para personalizar suas respostas e não repetir perguntas.
Aqui está a memória (talvez esteja vazia): {memory}

Seu papel é entender o momento do lead, educar sobre consórcio quando necessário e extrair, ao longo da conversa, as informações da memória.

Se você tiver informações técnicas relevantes sobre consórcios, use-as para responder às perguntas do usuário.
INFORMAÇÕES TÉCNICAS RELEVANTES:
{rag_context}

Use as informações técnicas acima quando forem relevantes para responder às perguntas do usuário sobre consórcios.
Se as informações técnicas não forem relevantes para a pergunta atual, ignore-as e responda naturalmente.

Foque em coletar os seguintes dados, naturalmente ao longo da conversa:
• [Necessidade principal]
• [Valor desejado do bem ou negócio]
• [Urgência do plano]
• [Nível de conhecimento sobre consórcios]
• [Disponibilidade de dar lance]
• [Finalidade: uso próprio ou investimento]
• [Orçamento mensal disponível]
• [Forma de tomada de decisão: forma de decidir, o que leva em consideração, como funciona o processo decisório]
"""


# Extraction instruction
TRUSTCALL_INSTRUCTION = TRUSTCALL_INSTRUCTION = """
Você é um agente responsável por atualizar a memória (JSON doc) do usuário com base na conversa abaixo.

Sua tarefa é revisar a conversa e preencher ou atualizar os seguintes campos no perfil do usuário:

- nome
- sobrenome
- email
- telefone
- necessidade
- valor_desejado
- urgencia
- nivel_conhecimento_consorcio
- disponibilidade_lance
- finalidade
- orcamento_mensal
- tomada_decisao

Responda com o objeto JSON `UserProfile` contendo os campos preenchidos ou atualizados a partir das mensagens. Ignore campos que não forem mencionados.
"""

# --------------------- CATEGORIA MAPPING ---------------------
categoria_map = {
    "Basico do Consórcio": ["consórcio", "inicial", "como funciona", "introdução"],
    "Dinamica de Grupos": ["grupo", "participante", "admissão", "fórum", "assembleia"],
    "Regras": ["regras", "normas", "contrato", "lei", "termo"],
    "Aspectos financeiros": [
        "valor",
        "preço",
        "custo",
        "parcela",
        "financiamento",
        "taxa",
    ],
    "Seguro e Responsabilidade": [
        "seguro",
        "responsabilidade",
        "garantia",
        "cobertura",
    ],
    "Contemplação e Lances": [
        "lance",
        "contemplação",
        "contemplado",
        "parcela",
        "oferta",
    ],
    "Uso de bens": [
        "uso",
        "bem",
        "propriedade",
        "bem imóvel",
        "uso do bem",
        "transferência",
    ],
}


def detectar_categoria(query: str) -> str:
    query = query.lower()
    for categoria, keywords in categoria_map.items():
        if any(keyword in query for keyword in keywords):
            return categoria
    return "Outros"


# --------------------- VETOR DB INSTANCE ---------------------
vector_db = SupabaseVectorDB()


# --------------------- RAG RETRIEVAL ---------------------
def get_rag_retrieval(query: str) -> str:
    try:
        hf = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        processed_query = query.strip().lower()
        query_embedding = hf.embed_query(processed_query)

        # Remover a lógica de detectar categoria e usar tags
        results = vector_db.search_similar_faqs(
            query_embedding=query_embedding, top_k=2
        )
        print(
            f"Resultados da busca: {results}"
        )  # Debug: Verificar os resultados da busca

        if not results:
            return "Nenhuma informação relevante encontrada."

        # Adaptando o formato de resposta
        response = [
            f"Q: {r['pergunta']}\nA: {r['resposta']}\nCategoria: {r['categoria']}"
            for r in results
        ]
        return "\n\n---\n\n".join(response)

    except Exception as e:
        return f"Erro ao buscar informações de suporte técnico: {e}"


# --------------------- CHATBOT NODE ---------------------
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    if existing_memory and existing_memory.value:
        mem = existing_memory.value
        formatted_memory = (
            f"Nome: {mem.get('nome', 'Desconhecido')}\n"
            f"Sobrenome: {mem.get('sobrenome', 'Desconhecido')}\n"
            f"E-mail: {mem.get('email', 'Desconhecido')}\n"
            f"Telefone: {mem.get('telefone', 'Desconhecido')}\n"
            f"Necessidade: {mem.get('necessidade', 'Desconhecida')}\n"
            f"Valor Desejado: {mem.get('valor_desejado', 'Desconhecido')}\n"
            f"Urgência: {mem.get('urgencia', 'Desconhecida')}\n"
            f"Nível de Conhecimento sobre Consórcio: {mem.get('nivel_conhecimento_consorcio', 'Desconhecido')}\n"
            f"Disponibilidade de Lance: {mem.get('disponibilidade_lance', 'Desconhecida')}\n"
            f"Finalidade: {mem.get('finalidade', 'Desconhecida')}\n"
            f"Orçamento Mensal: {mem.get('orcamento_mensal', 'Desconhecido')}\n"
            f"Tomada de Decisão: {mem.get('tomada_decisao', 'Desconhecida')}"
        )
    else:
        formatted_memory = "Nenhuma informação disponível ainda."

    user_message = state["messages"][-1].content
    rag_context = get_rag_retrieval(user_message)

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        memory=formatted_memory, rag_context=rag_context
    )
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}


# --------------------- MEMÓRIA NODE ---------------------
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("memory", user_id)

    existing_memory = store.get(namespace, "user_memory")
    existing_profile = (
        {"UserProfile": existing_memory.value} if existing_memory else None
    )

    result = trustcall_extractor.invoke(
        {
            "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]
            + state["messages"],
            "existing": existing_profile,
        }
    )

    updated_profile = result["responses"][0].model_dump()
    store.put(namespace, "user_memory", updated_profile)


# --------------------- GRAPH ---------------------
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()
