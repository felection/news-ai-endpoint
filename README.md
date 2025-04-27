# ml-api-service

## Overview
The `news-ai-endpoint` project provides a RESTful API for deploying machine learning models.
This service allows users to easily access and utilize pre-trained models for various natural language processing tasks.

## Features
- Deployment of Sentence Transformer models via a secure API.
- Header-based security for API access.
- Comprehensive documentation for API usage and model details.

## Project Structure
```
news-ai-endpoint/
├── src/                      # Main application code
│   ├── __init__.py
│   ├── main.py              # FastAPI app setup
│   ├── config.py            # Configuration settings
│   ├── dependencies.py      # Dependency injections
│   ├── models/             # Pydantic models
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── health.py
│   ├── routes/             # API endpoints
│   │   ├── __init__.py
│   │   ├── api.py          # Main router
│   │   ├── embedding.py    # Embedding endpoints
│   │   └── health.py       # Health/status endpoints
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   └── embedding.py    # Embedding service
│   └── utils/              # Utilities
│       ├── __init__.py
│       └── logging_utils.py     # Security utilities
│
├── tests/                    # Unit and integration tests (Recommended)
│   └── ...
│
├── .env                      # Environment variables (API keys, model lists) - **Gitignored!**
├── .env.example              # Example environment file
├── .gitignore                # Git ignore file
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
└── setup.py                  # 
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd news-ai-endpoint
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```

5. Set up environment variables by copying `.env.example` to `.env` and modifying as needed.

## Usage
To start the API service, run:
```
make run
```
The API will be available at `http://localhost:8000`.
The Documentation will be available at `http://localhost:8000/api/v1/docs`.
The health will be available at `http://localhost:8000/api/v1/health`.