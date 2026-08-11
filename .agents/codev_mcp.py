from fastmcp import FastMCP
import requests
import os
import google.generativeai as genai
from pathlib import Path

# Initialize FastMCP with a broader name to reflect its dual purpose
mcp = FastMCP("VRI_2026_AI_Router")

@mcp.tool()
def generate_verilog(prompt: str) -> str:
    """Uses the local CodeV model to generate Verilog code for the ZedBoard."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "hf.co/mradermacher/CodeV-DS-6.7B-GGUF:Q4_K_M",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "Error generating code.")
    return f"Ollama API Error: {response.status_code}"

@mcp.tool()
def ask_gemini_context(query: str, file_paths: list[str]) -> str:
    """
    Sends large files (.csv, .log, .cpp) or directories to Gemini for analysis.
    Use this to condense context before writing Python implementation logic.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "System Error: GEMINI_API_KEY environment variable is missing."
        
    genai.configure(api_key=api_key)
    
    # gemini-1.5-flash is ideal for high-speed, large-context retrieval tasks
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    compiled_context = []
    
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            compiled_context.append(f"[Warning: File not found - {path_str}]")
            continue
            
        try:
            # Read the raw file data
            content = path.read_text(encoding='utf-8')
            compiled_context.append(f"--- START FILE: {path_str} ---\n{content}\n--- END FILE: {path_str} ---")
        except Exception as e:
            compiled_context.append(f"[Error reading {path_str}: {str(e)}]")
            
    # Construct the final prompt for Gemini
    full_prompt = (
        f"You are the Researcher for the VRI 2026 project. Analyze the provided context.\n\n"
        f"Query: {query}\n\n"
        f"Context Files:\n" + "\n".join(compiled_context)
    )
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()