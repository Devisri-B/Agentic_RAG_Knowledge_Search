<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic RAG Knowledge Search</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            border-bottom: 2px solid #eaeaea;
            padding-bottom: 0.5em;
        }
        code {
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            overflow: auto;
            border-radius: 6px;
            line-height: 1.45;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            font-size: 100%;
        }
        ul {
            padding-left: 2em;
        }
        li {
            margin-bottom: 0.5em;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>

    <h1>Agentic RAG Knowledge Search</h1>

    <p>A production-ready AI Microservice that uses an <strong>Agentic Router</strong> to intelligently decide between retrieving answers from internal private documents (RAG) or searching the live web.</p>

    <p>Built with <strong>FastAPI</strong>, <strong>LangChain</strong>, <strong>Docker</strong>, and <strong>Google Gemini</strong>.</p>

    <h2> Features</h2>
    <ul>
        <li><strong>Agentic Routing</strong>: Uses an LLM to decide <em>where</em> to get information.</li>
        <li><strong>Hybrid Tools</strong>:
            <ul>
                <li><code>lookup_internal_policy</code>: Searches local PDFs using FAISS + Embeddings.</li>
                <li><code>search_web</code>: Searches DuckDuckGo for real-time info.</li>
            </ul>
        </li>
        <li><strong>REST API</strong>: Fully documented API using FastAPI.</li>
        <li><strong>Containerized</strong>: Docker support for easy deployment.</li>
    </ul>

    <h2> Project Structure</h2>
    <ul>
        <li><code>src/rag_engine.py</code>: Handles PDF ingestion and Vector Database (FAISS).</li>
        <li><code>src/agent.py</code>: Defines the Agent, Tools, and LangChain logic.</li>
        <li><code>src/main.py</code>: The FastAPI server entry point.</li>
        <li><code>data/</code>: Place your PDF documents here.</li>
    </ul>

    <h2> Setup & Installation</h2>

    <h3>1. Prerequisites</h3>
    <ul>
        <li>Get a <a href="https://aistudio.google.com/">Free Google Gemini API Key</a>.</li>
        <li>Create a <code>.env</code> file in this directory:
            <pre><code>GOOGLE_API_KEY=AIzaSyD...your_key_here...</code></pre>
        </li>
    </ul>

    <h3>2. Add Data</h3>
    <p>Place your PDF file (e.g., policy documents, course materials) into the <code>data/</code> folder and rename it to <code>policy.pdf</code> (or update <code>src/agent.py</code> to match your filename).</p>

    <h3>3. Run with Docker (Recommended)</h3>
    <pre><code>docker build -t agentic-rag .
docker run -p 8000:8000 --env-file .env agentic-rag</code></pre>

    <h3>4. Run Locally</h3>
    <pre><code>pip install -r requirements.txt
python src/main.py</code></pre>

    <h2> API Usage</h2>
    <p><strong>Endpoint:</strong> <code>POST /chat</code></p>

    <pre><code>{
  "query": "What are the rules for termination in the policy?"
}</code></pre>

    <p><strong>Response:</strong></p>
    <pre><code>{
  "response": "According to the policy, termination requires...",
  "steps": [
    {
      "tool": "lookup_internal_policy",
      "tool_input": "termination rules"
    }
  ]
}</code></pre>

</body>
</html>