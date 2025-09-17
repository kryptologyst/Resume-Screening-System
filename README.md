# Resume Screening System

A sophisticated AI-powered resume screening system that uses state-of-the-art NLP techniques to match candidates with job descriptions. Built with BERT embeddings, TF-IDF analysis, and advanced similarity metrics.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Advanced NLP Techniques
- **BERT Embeddings**: Uses SentenceTransformers for semantic understanding
- **TF-IDF Analysis**: Traditional text vectorization for keyword matching
- **Multi-dimensional Scoring**: Combines semantic similarity, skill matching, and experience weighting
- **Configurable Weights**: Adjust importance of different scoring metrics

### Modern Web Interface
- **Interactive Dashboard**: Built with Streamlit for intuitive user experience
- **Real-time Analytics**: Visualizations with Plotly and interactive charts
- **Resume Management**: Upload, view, and manage candidate resumes
- **Job Management**: Create and manage job descriptions
- **Screening Results**: Detailed candidate rankings with score breakdowns

### Database Integration
- **SQLite Database**: Persistent storage for resumes, jobs, and screening results
- **Mock Data Generator**: Realistic sample data for testing and demonstration
- **Data Analytics**: Insights into candidate pools and skill distributions

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd 0069_Resume_screening_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model (optional, for advanced text processing):
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Generate Mock Data
```python
python mock_data.py
```

### 2. Run the Web Interface
```bash
streamlit run streamlit_app.py
```

### 3. Use the Command Line Interface
```python
python resume_screener.py
```

## Usage Examples

### Basic Resume Screening
```python
from resume_screener import ResumeScreener

# Initialize the screener
screener = ResumeScreener()

# Add a job description
job_id = screener.add_job_description(
    title="Senior Python Developer",
    content="Looking for a Python developer with ML experience...",
    required_skills=["Python", "Django", "scikit-learn"],
    experience_required=3
)

# Add resumes
screener.add_resume("John Doe", "Experienced Python developer...")

# Screen resumes
results = screener.screen_resumes(job_id)

# Display results
for result in results:
    print(f"Rank #{result['ranking']}: {result['resume_name']}")
    print(f"Score: {result['final_score']:.3f}")
```

### Custom Scoring Weights
```python
# Define custom weights for scoring metrics
custom_weights = {
    'bert': 0.5,      # Semantic similarity
    'tfidf': 0.2,     # Keyword matching
    'skills': 0.2,    # Skill overlap
    'experience': 0.1  # Experience matching
}

results = screener.screen_resumes(job_id, weights=custom_weights)
```

## Architecture

### Core Components

1. **ResumeScreener**: Main class handling all screening operations
2. **Database Layer**: SQLite for persistent data storage
3. **NLP Pipeline**: BERT + TF-IDF for text analysis
4. **Scoring Engine**: Multi-metric candidate evaluation
5. **Web Interface**: Streamlit-based user interface

### Scoring Methodology

The system uses a weighted combination of four metrics:

- **BERT Score (40%)**: Semantic similarity using sentence transformers
- **TF-IDF Score (30%)**: Traditional keyword-based matching
- **Skill Match (20%)**: Jaccard similarity of required vs. candidate skills
- **Experience Score (10%)**: Normalized experience level matching

### Database Schema

```sql
-- Resumes table
CREATE TABLE resumes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    skills TEXT,
    experience_years INTEGER,
    created_at TIMESTAMP
);

-- Job descriptions table
CREATE TABLE job_descriptions (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    required_skills TEXT,
    experience_required INTEGER,
    created_at TIMESTAMP
);

-- Screening results table
CREATE TABLE screening_results (
    id INTEGER PRIMARY KEY,
    job_id INTEGER,
    resume_id INTEGER,
    final_score REAL,
    bert_score REAL,
    tfidf_score REAL,
    ranking INTEGER,
    created_at TIMESTAMP
);
```

## Web Interface Features

### Dashboard
- Overview metrics and statistics
- Skills word cloud visualization
- Recent activity tracking

### Resume Screening
- Job selection interface
- Configurable scoring weights
- Interactive results visualization
- Multi-dimensional candidate comparison

### Data Management
- Resume upload and management
- Job description creation
- Bulk data operations

### Analytics
- Experience distribution analysis
- Skills frequency analysis
- Resume length statistics
- Interactive charts and graphs

## 🔧 Configuration

### Model Configuration
```python
# Initialize with different BERT model
screener = ResumeScreener(model_name='all-mpnet-base-v2')

# Custom TF-IDF parameters
screener.tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 3)
)
```

### Database Configuration
```python
# Custom database path
screener = ResumeScreener()
screener.db_path = 'custom_database.db'
```

## Testing

Run the example screening:
```bash
python resume_screener.py
```

Generate and test with mock data:
```bash
python mock_data.py
```

## Dependencies

### Core Libraries
- `sentence-transformers`: BERT embeddings
- `scikit-learn`: TF-IDF and similarity metrics
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `sqlite3`: Database operations

### Web Interface
- `streamlit`: Web application framework
- `plotly`: Interactive visualizations
- `wordcloud`: Text visualization

### Optional
- `spacy`: Advanced NLP processing
- `nltk`: Natural language toolkit
- `python-docx`: Word document parsing
- `PyPDF2`: PDF parsing

## Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment
1. Set up a virtual environment
2. Install dependencies
3. Configure database path
4. Deploy using your preferred platform (Heroku, AWS, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] PDF/Word document parsing
- [ ] Advanced skill extraction using NER
- [ ] Machine learning model for custom scoring
- [ ] Integration with job boards APIs
- [ ] Email notification system
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] REST API endpoints

## Support

For questions, issues, or contributions, please open an issue on GitHub.


# Resume-Screening-System
