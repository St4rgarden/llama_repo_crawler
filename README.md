# GitHub Issue Analyzer

This project is a Python-based tool that analyzes GitHub repositories using natural language processing and machine learning techniques. It loads from specified GitHub repositories, creates embeddings, and allows for querying the data using a question-answering system.

## Features

- Load issues from up to three GitHub repositories
- Create embeddings using VoyageAI's `voyage-code-2` model
- Use FAISS for efficient similarity search
- Implement a question-answering system using OpenAI's language model

## Prerequisites

- Python 3.8+
- A GitHub personal access token
- VoyageAI API key
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/github-issue-analyzer.git
   cd github-issue-analyzer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root directory with the following content:
   ```
   GITHUB_TOKEN=your_github_personal_access_token
   VOYAGE_API_KEY=your_voyage_api_key
   OPENAI_API_KEY=your_openai_api_key

   # Repository 1
   GITHUB_OWNER_1=owner1
   GITHUB_REPO_1=repo1
   GITHUB_BRANCH_1=main

   # Repository 2
   GITHUB_OWNER_2=owner2
   GITHUB_REPO_2=repo2
   GITHUB_BRANCH_2=main

   # Repository 3
   GITHUB_OWNER_3=owner3
   GITHUB_REPO_3=repo3
   GITHUB_BRANCH_3=main
   ```

   Replace the placeholder values with your actual API keys and repository details.

## Usage

Run the main script:

```
python main.py
```

The script will:
1. Load issues from the specified GitHub repositories
2. Create embeddings and a vector store
3. Set up a question-answering system
4. Run an example query

You can modify the example query in `main.py` to ask different questions about the loaded issues.

## Project Structure

- `main.py`: The main script that orchestrates the entire process
- `docs.py`: Handles loading documents from GitHub repositories
- `embeddings.py`: Creates embeddings and the vector store
- `helpers.py`: Contains utility functions
- `requirements.txt`: Lists all the Python dependencies

## Customization

- To analyze different repositories, update the repository details in the `.env` file
- To modify the embedding process, edit the `embeddings.py` file
- To change how documents are loaded or processed, edit the `docs.py` file

## Troubleshooting

If you encounter any issues:
1. Ensure all API keys are correctly set in the `.env` file
2. Check that you have the necessary permissions for the GitHub repositories
3. Verify that all dependencies are installed correctly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.