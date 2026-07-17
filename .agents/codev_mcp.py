from fastmcp import FastMCP
import requests

mcp = FastMCP("CodeV_Verilog_Generator")

@mcp.tool()
def generate_verilog(prompt: str) -> str:
    """Uses the local CodeV model to generate Verilog code."""
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

if __name__ == "__main__":
    mcp.run()