import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

# Carregar variáveis de ambiente
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")


class SupabaseVectorDB:
    def __init__(self):
        self._connection = None
        self._connect()

    def _connect(self):
        """Estabelece conexão com o banco."""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(
                    host=DB_HOST,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    port=DB_PORT,
                )
                self._connection.autocommit = True
            except Exception as e:
                raise ConnectionError(f"Erro ao conectar ao banco: {e}")

    def close(self):
        """Fecha conexão com o banco."""
        if self._connection and not self._connection.closed:
            self._connection.close()

    def _format_embedding(self, embedding: List[float]) -> str:
        """Formata lista de floats para o tipo vector do PostgreSQL."""
        return "ARRAY[%s]" % ", ".join(map(str, embedding))  # Convertendo para array

    def search_similar_faqs(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        categoria: Optional[str] = None,
        tags: Optional[List[str]] = None,
        similarity_threshold: float = 0.6,
    ) -> List[dict]:
        """
        Busca as FAQs mais semelhantes usando pgvector.
        Filtros por categoria e tags são opcionais.
        """
        self._connect()
        formatted_embedding = self._format_embedding(query_embedding)
        filters = []
        params = []

        if categoria:
            filters.append("metadata->>'categoria' = %s")
            params.append(categoria)

        if tags:
            filters.append(
                "metadata->'tags' ?| array[%s]" % ", ".join(["%s"] * len(tags))
            )
            params.extend(tags)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"""
            SELECT
                id,
                pergunta,
                resposta,
                metadata->>'categoria' AS categoria,
                metadata,
                1 - (embedding <=> {formatted_embedding}::vector) AS similaridade
            FROM faq_embeddings
            {where_clause}
            HAVING 1 - (embedding <=> {formatted_embedding}::vector) >= %s
            ORDER BY similaridade DESC
            LIMIT %s;
        """
        params.append(
            similarity_threshold
        )  # Adiciona o limiar diretamente nos parâmetros da consulta
        params.append(top_k * 2)  # Pega mais do que precisa, para filtrar depois

        try:
            with self._connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
        except Exception as e:
            print(f"Erro ao executar a consulta: {e}")
            return []

        return results[:top_k]

    def __del__(self):
        self.close()
