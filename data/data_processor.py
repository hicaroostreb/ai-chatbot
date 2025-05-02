import json
import os
import psycopg2
from psycopg2 import extras, extensions
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()

# Informações de conexão com o banco PostgreSQL (Supabase)
POSTGRES_HOST = os.getenv("SUPABASE_DB_HOST")
POSTGRES_PORT = int(os.getenv("SUPABASE_DB_PORT"))
POSTGRES_DB = os.getenv("SUPABASE_DB_NAME")
POSTGRES_USER = os.getenv("SUPABASE_DB_USER")
POSTGRES_PASS = os.getenv("SUPABASE_DB_PASSWORD")

# Caminho do JSON com os dados
JSON_FILE_PATH = "S:/Code/LangGraph_study/HandsOn/data/faq.json"

# Modelo E5-base multilíngue
MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIMENSION = 768  # Dimensão dos embeddings para o modelo E5-base

# Carregar o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def create_table(conn):
    """Criar a tabela faq_embeddings se não existir (versão local)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faq_embeddings (
            id TEXT PRIMARY KEY,
            pergunta TEXT,
            resposta TEXT,
            categoria TEXT,
            palavras_chave TEXT[],
            perguntas_relacionadas TEXT[],
            embedding VECTOR(768),
            metadata JSONB
        );
        """
    )
    conn.commit()
    cur.close()


def connect_to_postgres():
    """Conectar ao PostgreSQL e retornar a conexão."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
    )
    create_table(conn)
    return conn


def get_embedding_from_model(text: str) -> list:
    """Gerar embedding usando o modelo E5 com prefixo correto e normalização."""
    prefixed_text = f"passage: {text.strip()}"
    inputs = tokenizer(
        prefixed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token

    # Normalizar o vetor (como Langchain faz)
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.squeeze().tolist()


def insert_or_update_embedding_row(
    cur,
    id_faq,
    pergunta,
    resposta,
    categoria,
    palavras_chave,
    perguntas_relacionadas,
    embedding_vector,
    metadata,  # novo campo
):
    """Inserir ou atualizar uma linha na tabela faq_embeddings."""
    check_query = "SELECT id FROM faq_embeddings WHERE id = %s"
    cur.execute(check_query, (id_faq,))
    existing_item = cur.fetchone()

    if existing_item:
        update_query = """
            UPDATE faq_embeddings
            SET pergunta = %s, resposta = %s, categoria = %s, palavras_chave = %s, perguntas_relacionadas = %s, embedding = %s, metadata = %s
            WHERE id = %s
        """
        cur.execute(
            update_query,
            (
                pergunta,
                resposta,
                categoria,
                palavras_chave,
                perguntas_relacionadas,
                embedding_vector,
                json.dumps(metadata),
                id_faq,
            ),
        )
    else:
        insert_query = """
            INSERT INTO faq_embeddings (id, pergunta, resposta, categoria, palavras_chave, perguntas_relacionadas, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            insert_query,
            (
                id_faq,
                pergunta,
                resposta,
                categoria,
                palavras_chave,
                perguntas_relacionadas,
                embedding_vector,
                json.dumps(metadata),
            ),
        )


def process_json_and_store_embeddings(json_file_path):
    """Processar o JSON e armazenar embeddings no banco."""
    conn = connect_to_postgres()
    cur = conn.cursor()

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                try:
                    id_faq = item["id"]
                    pergunta = item["pergunta_principal"]
                    resposta = item["resposta"]
                    categoria = item["categoria"]
                    palavras_chave = item["palavras_chave"]
                    perguntas_relacionadas = (
                        item.get("perguntas_relacionadas", []) or []
                    )
                    embedding_input = item.get("embedding_input", "").strip()

                    # Validar se o campo embedding_input não está vazio
                    if not embedding_input:
                        print(f"FAQ {id_faq} ignorado por falta de texto de embedding.")
                        continue

                    # Gerar embedding
                    embedding_vector = get_embedding_from_model(embedding_input)

                    # Criar metadata estruturado
                    metadata = {
                        "categoria": categoria,
                        "tags": palavras_chave,
                        "origem": "faq",
                        "idioma": "pt",
                    }

                    # Inserir ou atualizar no banco
                    insert_or_update_embedding_row(
                        cur,
                        id_faq,
                        pergunta,
                        resposta,
                        categoria,
                        palavras_chave,
                        perguntas_relacionadas,
                        embedding_vector,
                        metadata,
                    )
                except Exception as e:
                    print(f"Erro ao processar item {item.get('id')}: {e}")
                    continue

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Erro geral ao processar os dados: {e}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    process_json_and_store_embeddings(JSON_FILE_PATH)
    print("Dados processados e embeddings armazenados com sucesso.")
