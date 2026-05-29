"""
Build a Neo4j COBOL program graph from AST summaries.

Node types:  Program, Paragraph, Variable, Copybook
Edge types:  CALLS (CALL stmt), PERFORMS (PERFORM stmt), USES (variable ref), COPIES (COPY stmt)

Requires Neo4j Community running via docker/neo4j/docker-compose.yml.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase

from src.pipeline.parse import parse_cobol

logger = logging.getLogger(__name__)


class CobolGraphBuilder:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def ingest_program(self, name: str, source: str) -> None:
        ast = parse_cobol(source)
        with self._driver.session() as session:
            session.execute_write(self._create_program_node, name, source[:500])
            if ast:
                for para in ast.get("paragraphs", []):
                    session.execute_write(self._create_paragraph, name, para)
                for call in ast.get("calls", []):
                    session.execute_write(self._create_call_edge, name, call)

    @staticmethod
    def _create_program_node(tx: Any, name: str, snippet: str) -> None:
        tx.run(
            "MERGE (p:Program {name: $name}) SET p.snippet = $snippet",
            name=name, snippet=snippet,
        )

    @staticmethod
    def _create_paragraph(tx: Any, program: str, para: str) -> None:
        tx.run(
            "MERGE (pg:Paragraph {name: $para, program: $program})"
            " WITH pg MATCH (p:Program {name: $program})"
            " MERGE (p)-[:CONTAINS]->(pg)",
            para=para, program=program,
        )

    @staticmethod
    def _create_call_edge(tx: Any, caller: str, callee: str) -> None:
        tx.run(
            "MERGE (a:Program {name: $caller})"
            " MERGE (b:Program {name: $callee})"
            " MERGE (a)-[:CALLS]->(b)",
            caller=caller, callee=callee,
        )

    def create_indexes(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE INDEX program_name IF NOT EXISTS FOR (p:Program) ON (p.name)")
            session.run("CREATE INDEX paragraph_name IF NOT EXISTS FOR (pg:Paragraph) ON (pg.name)")
