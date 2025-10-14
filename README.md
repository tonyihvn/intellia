# MCP OpenMRS Application

This project implements a Model Context Protocol (MCP) to connect to an OPENMRS MySQL database, enabling the application to answer natural language questions by understanding the database schema and querying it. The application is designed to be hosted locally and utilizes a local LLM (such as CodeLlama) for processing natural language queries. The web interface is built using Flask and Streamlit.

## Project Structure

```
mcp-openmrs-app
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── streamlit_app.py
│   ├── core
│   │   ├── __init__.py
│   │   └── query_handler.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── schema.py
│   ├── llm
│   │   ├── __init__.py
│   │   └── client.py
│   └── web
│       ├── __init__.py
│       ├── routes.py
│       └── templates
│           └── index.html
├── tests
│   ├── __init__.py
│   ├── test_core.py
│   └── test_db.py
├── config.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd mcp-openmrs-app
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Configure the application:**
   Update the `config.py` file with your database connection parameters and LLM settings.

4. **Run the Flask application:**
   Start the Flask application by executing:
   ```
   python src/main.py
   ```

5. **Run the Streamlit application:**
   Start the Streamlit application by executing:
   ```
   streamlit run src/streamlit_app.py
   ```

6. **Access the applications:**
   - For the Flask application, open your web browser and navigate to `http://localhost:5000`.
   - For the Streamlit application, open your web browser and navigate to `http://localhost:8501`.

## Usage

Once the applications are running, you can input natural language questions related to the OPENMRS database. The applications will process these questions, query the database, and return the relevant answers.

## MCP Functionality

The Model Context Protocol (MCP) allows the applications to understand the context of the questions being asked and formulate appropriate SQL queries based on the database schema. This enables dynamic interaction with the database, providing users with accurate and relevant information.

## Testing

Unit tests are provided in the `tests` directory to ensure the core functionality and database interactions work as expected. You can run the tests using:
```
pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.