import json
import ollama
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from rag.query import (
    retrieve,
    build_prompt,
    ask_llm,
    prepare_contexts,
    existence_listing_answer,
    inventory_overview_answer,
    format_corpus_overview,
    REFUSE_ANSWER,
)
from mcp.client import MCPClient
from config import OLLAMA_MODEL


def mcp_tool_text(payload) -> str:
    """Unwrap FastMCP / JSON-RPC tools/call payload to plain text."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return str(payload)

    # Full JSON-RPC envelope
    if "jsonrpc" in payload and "result" in payload:
        return mcp_tool_text(payload.get("result"))

    structured = payload.get("structuredContent")
    if isinstance(structured, dict) and structured.get("result") is not None:
        value = structured["result"]
        return value if isinstance(value, str) else str(value)

    content = payload.get("content")
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        if parts:
            return "\n".join(parts)

    nested = payload.get("result")
    if isinstance(nested, str):
        return nested
    if isinstance(nested, dict):
        return mcp_tool_text(nested)

    return ""


class CompanyKBAssistant:
    """Company Knowledge Base Assistant combining RAG and MCP."""
    
    def __init__(self):
        self.llm_client = ollama.Client()
        self.mcp = None
        self._init_mcp()
        
    def _init_mcp(self):
        """Initialize MCP client."""
        try:
            import sys
            from pathlib import Path
            python_cmd = sys.executable
            # Get absolute path to MCP server
            mcp_path = Path(__file__).parent / "mcp" / "server.py"
            self.mcp = MCPClient([python_cmd, str(mcp_path)])
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}")
            self.mcp = None
    
    def _mcp_decision_prompt(self, query: str, contexts) -> str:
        """Build the MCP tool-routing prompt (paths must come from retrieved sources)."""
        source_paths = []
        seen = set()
        for ctx in contexts or []:
            src = ctx.get("source")
            if src and src not in seen:
                seen.add(src)
                source_paths.append(src)

        if contexts:
            context_summary = (
                f"Retrieved {len(contexts)} relevant chunks from knowledge base:\n"
            )
            for i, ctx in enumerate(contexts[:3], 1):
                context_summary += (
                    f"{i}. From {ctx['source']}: {ctx['text'][:200]}...\n"
                )
        else:
            context_summary = "No relevant chunks found in knowledge base.\n"

        if source_paths:
            paths_block = "Available source paths (copy exactly if using read_document):\n" + "\n".join(
                f"- {p}" for p in source_paths
            )
            example_path = source_paths[0]
        else:
            paths_block = "Available source paths: (none — do not call read_document)"
            example_path = "docs/vacation-policy.md"

        return f"""You are helping answer a question using a knowledge base system with RAG (retrieval) and MCP tools.

User question: {query}

{context_summary}
{paths_block}

Available MCP tools:
1. read_document(file_path: str) - Read a specific document file (use when you need full document content)
2. list_documents() - List all available documents (use when user asks "what documents exist" or "list all docs")
3. search_documents(query: str) - Search for documents by name (use when user asks to find a specific document)

Decision rules:
- If the user asks what documents/topics/info the knowledge base covers, or how to use the assistant, use list_documents
- If the retrieved chunks fully answer the question, set use_mcp to false
- If chunks are empty or insufficient, consider using MCP tools
- If user explicitly asks to read/list/search documents, use the appropriate tool
- If you need the full content of a specific document mentioned in chunks, use read_document
- For read_document, file_path MUST be copied EXACTLY from "Available source paths" above. Never invent or shorten paths (e.g. do not invent docs/asyncpg.md).
- When in doubt on topical questions, prefer not using MCP (chunks are usually sufficient)

Respond ONLY with valid JSON, no other text:
{{"use_mcp": true/false, "tool": "tool_name_or_null", "args": {{"arg_name": "value"}}}}

Examples:
{{"use_mcp": false, "tool": null, "args": {{}}}}
{{"use_mcp": true, "tool": "read_document", "args": {{"file_path": "{example_path}"}}}}
{{"use_mcp": true, "tool": "list_documents", "args": {{}}}}
{{"use_mcp": true, "tool": "search_documents", "args": {{"query": "vacation"}}}}

Your JSON response:"""

    def _llm_decide_mcp_usage(self, query: str, contexts):
        """Ask LLM if MCP tools are needed based on query and retrieved contexts."""
        if not self.mcp:
            return None, None

        decision_prompt = self._mcp_decision_prompt(query, contexts)
        allowed_paths = {
            ctx.get("source") for ctx in (contexts or []) if ctx.get("source")
        }

        try:
            response = self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that decides when to use tools. Always respond with valid JSON only."},
                    {"role": "user", "content": decision_prompt}
                ]
            )
            
            response_text = response["message"]["content"].strip()
            
            # Clean up JSON if wrapped in markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            decision = json.loads(response_text)
            
            if decision.get("use_mcp", False):
                tool_name = decision.get("tool")
                tool_args = decision.get("args", {}) or {}
                if tool_name == "read_document":
                    path = tool_args.get("file_path")
                    if path not in allowed_paths:
                        # Invented / shortened paths are rejected; answer from chunks.
                        return None, None
                return tool_name, tool_args
            
            return None, None
            
        except Exception as e:
            # If LLM decision fails, don't use MCP
            return None, None
    
    def _call_mcp_tool(self, tool_name: str, tool_args: dict):
        """Call an MCP tool with given name and arguments."""
        if not self.mcp:
            return None
        
        try:
            result = self.mcp.call_tool(tool_name, tool_args)
            return mcp_tool_text(result)
        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {str(e)}"
    
    def query(self, user_query: str, verbose=False):
        """Answer a question using RAG and optionally MCP tools."""
        inventory = inventory_overview_answer(user_query)
        if inventory is not None:
            answer, contexts = inventory
            if verbose:
                print("📚 Answering from knowledge-base document inventory")
            return {
                "answer": answer,
                "sources": [c["source"] for c in contexts] if contexts else [],
                "mcp_used": False,
                "mcp_tool": None,
            }

        contexts = prepare_contexts(user_query, retrieve(user_query))

        if verbose:
            print(f"📚 Retrieved {len(contexts)} relevant chunks from knowledge base")

        listing_answer = existence_listing_answer(user_query, contexts)
        if listing_answer is not None:
            return {
                "answer": listing_answer,
                "sources": [c["source"] for c in contexts] if contexts else [],
                "mcp_used": False,
                "mcp_tool": None,
            }

        mcp_result = None
        mcp_tool_used = None
        tool_name, tool_args = self._llm_decide_mcp_usage(user_query, contexts)

        if tool_name:
            if verbose:
                print(f"🔧 LLM decided to use MCP tool: {tool_name} with args: {tool_args}")
            mcp_result = self._call_mcp_tool(tool_name, tool_args)
            # Failed / invented paths must not poison generation when RAG already has facts.
            if mcp_result and mcp_result.startswith("Error") and contexts:
                if verbose:
                    print(
                        f"⚠️  MCP tool failed ({mcp_result[:80]}); "
                        "answering from retrieved chunks instead"
                    )
                mcp_result = None
            else:
                mcp_tool_used = tool_name
                if verbose and mcp_result:
                    print(f"✅ MCP tool returned result (length: {len(mcp_result)} chars)")

        # list_documents + empty RAG must not hit the refuse-only prompt.
        if (
            mcp_tool_used == "list_documents"
            and mcp_result
            and not str(mcp_result).startswith("Error")
        ):
            answer, inv_contexts = format_corpus_overview(howto=True)
            return {
                "answer": answer,
                "sources": [c["source"] for c in inv_contexts],
                "mcp_used": True,
                "mcp_tool": mcp_tool_used,
            }

        gen_contexts = list(contexts) if contexts else []
        if mcp_result and not gen_contexts and not str(mcp_result).startswith("Error"):
            gen_contexts = [
                {
                    "source": f"mcp:{mcp_tool_used}",
                    "chunk_id": 0,
                    "text": mcp_result,
                }
            ]

        prompt = build_prompt(user_query, gen_contexts)
        if mcp_result and contexts:
            injection = (
                f"\n\n<additional_info_from_mcp_tool>\n{mcp_result}\n"
                f"</additional_info_from_mcp_tool>\n"
            )
            marker = "<assistant>"
            if marker in prompt:
                prompt = prompt.replace(marker, injection + marker, 1)
            else:
                prompt += injection

        answer = ask_llm(prompt)
        sources = [c["source"] for c in contexts] if contexts else [
            c["source"] for c in gen_contexts
        ]

        return {
            "answer": answer,
            "sources": sources,
            "mcp_used": mcp_result is not None,
            "mcp_tool": mcp_tool_used
        }
    
    def close(self):
        """Clean up resources."""
        if self.mcp:
            self.mcp.close()


if __name__ == "__main__":
    assistant = CompanyKBAssistant()
    
    print("🤖 Company Knowledge Base Assistant")
    print("Type 'exit' or 'quit' to stop\n")
    
    try:
        while True:
            query = input("❓ Question: ")
            if query.lower() in {"exit", "quit"}:
                break
            
            print("\n" + "─" * 60)
            result = assistant.query(query, verbose=True)
            
            print("\n🤖 Answer:\n")

            console = Console(force_terminal=True)
            console.print(Markdown(result["answer"]))
            
            if result["sources"] and result["answer"].strip() != REFUSE_ANSWER:
                print("\n📚 Sources:")
                seen_sources = set()
                for src in result["sources"]:
                    if src not in seen_sources:
                        print(f"  • {src}")
                        seen_sources.add(src)
            
            if result["mcp_used"]:
                print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")
            
            print("─" * 60 + "\n")
    
    finally:
        assistant.close()
