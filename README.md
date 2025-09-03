# AI-Powered Email Copilot for Enterprise Support

A multi-agent AI system that empowers company engineers by enabling them to interact with a chatbot for insights about projects and clients. The chatbot’s knowledge base is dynamically built from filtered email communications, chunked and structured for intelligent retrieval.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Features
- Chatbot interface for engineers to query about projects or clients.
- Real-time knowledge base sourced from company mailbox.
- Intelligent email filtering using Groq to identify important communications.
- Emails are chunked with langchain-experimental and structured with Agno.
- Knowledge graph construction and advanced retrieval using Neo4j, LightRAG, and rag-anything.
- Secure integration with Gmail API via Google Cloud Console.
- Uses Ollama as the core LLM for advanced reasoning.
- Docker support for simplified deployment.

## Architecture
- **Frontend:** Angular app for engineers to interact with the chatbot.
- **Backend:** FastAPI server orchestrates multi-agent logic and connections.
- **Groq Filter:** Filters emails, forwarding only relevant ones.
- **Chunking:** Emails are chunked with langchain-experimental.
- **Structuring:** Emails are structured with Agno.
- **Knowledge Graph:** Neo4j stores structured project/client information; LightRAG and rag-anything enrich context and retrieval.
- **Ollama:** LLM for reasoning and answering queries.

## Tech Stack
- **Frontend:** Angular
- **Backend:** FastAPI (Python)
- **Email Integration:**Gmail API (Google Cloud Console)
- **Filtering:** Groq
- **Chunking:** langchain-experimental
- **Structuring:** Agno
- **Knowledge Graph:** Neo4j (graph database), LightRAG & rag-anything (retrieval & context enrichment)
- **LLM:** Ollama
- **Containerization:** Docker

## Installation

### Prerequisites
- Node.js & npm (for Angular frontend)
- Python 3.8+ (for FastAPI backend)
- Docker (for containerized setup)
- Access to Google Cloud Console (Gmail API setup)
- Neo4j instance (can be run in Docker)
- Ollama model installed
- Install langchain-experimental, Agno, lightrag, rag-anything (see requirements.txt or docs)

### Clone the repository
```bash
git clone https://github.com/ahmedkbenzid/AI-Powered-Email-Copilot-for-Enterprise-Support.git
cd AI-Powered-Email-Copilot-for-Enterprise-Support
```

### Docker Setup

1. Build and run the containers:
    ```bash
    docker-compose up --build
    ```
2. This will start the frontend, backend, Neo4j, and other necessary services.

### Manual Setup (Alternative to Docker)

#### Backend Setup
1. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure environment variables (see [Configuration](#configuration)).
4. Run backend server:
    ```bash
    uvicorn main:app --reload
    ```

#### Frontend Setup
1. Navigate to the frontend directory (if applicable):
    ```bash
    cd frontend
    ```
2. Install dependencies:
    ```bash
    npm install
    ```
3. Run Angular app:
    ```bash
    ng serve
    ```

## Configuration

- **Gmail API:** Set up credentials in Google Cloud Console and save them as `credentials.json` in your backend directory.
- **Neo4j:** Configure connection strings in your environment variables. If using Docker, these are set in `docker-compose.yml`.
- **Ollama:** Make sure the LLM is installed and accessible.
- **Environment Variables:** Create a `.env` file with settings for Gmail, Neo4j, LLM endpoints, etc.
- **Knowledge Base:** Ensure langchain-experimental, agno, lightrag, and rag-anything are properly configured.

## Usage

- Start both frontend and backend servers (or run `docker-compose up`).
- Log in as an engineer.
- Interact with the chatbot to ask questions about projects or clients.
- The system will fetch, filter, and process emails, chunking them with langchain-experimental, structuring them with agno, storing them in Neo4j, and using LightRAG and rag-anything for knowledge graph construction and retrieval, enabling smart responses.

## Contributing

Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.


## Contact

For questions or support, please open an issue or contact [ahmedkbenzid](https://github.com/ahmedkbenzid).
