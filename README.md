# Real-time RAG Local Chatbot
## Learning Project
A simple real-time RAG Local Chatbot, using a small and open source LLM.
An exploratory alternative over cloud-based solutions in terms of privacy, security and operational independence.

## Table of Contents
- [Run Locally](#run-locally)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Next Steps](#next-steps)
- [License](#license)
    
## Run Locally
- Requires uv (package and project manager)
- Requires to configure the .env variables. Copy the .example.env to .env.

Clone the project

```bash
  git clone https://github.com/josezg/real-time-RAG-chatbot
```

Go to the project directory

```bash
  cd real-time-RAG-chatbot
```

Install dependencies

```bash
  uv sync
```

Start the server

```bash
  streamlit run src/app.py
```

## Features
- Input a directory path. (It must have a documents sub-folder with the documents intended to query with the RAG)
- Process the documents, generates embeddings and create a ChromaDB.
- Query the documents through the RAG Chatbot.
- Show the answer in the UI.
- Allows the user to re-index the documents when the synced directory files have changed.
- Allows the user to clear the chat history in the UI.

## Tech Stack

- Streamlit for the web interface
- BAAI/bge-small-en-v1.5 model for embeddings
- microsoft/phi-2 as generative LLM.
- LlamaIndex for the main logic
- uv as the python dependencies manager.

## Next steps
This project aims to continuously improve and expand its capabilities, as a learning project. Some potential future developments include:

- Optimization in the RAG techniques.
- Selection of the best small open source LLM.
- Add a system prompt to have a better answer generation.
- Test more file types as documents and data sources.
- Improve the logic and flow in the UI.
- Test the RAG with more complex documents.
- Code logic optimizations, improvements and bugs fixing.

## License
[MIT](https://choosealicense.com/licenses/mit/)