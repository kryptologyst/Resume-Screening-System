# 🚀 Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers
```

### 2. Generate Sample Data
```bash
python3 mock_data.py
```

### 3. Run Web Interface
```bash
streamlit run streamlit_simple.py
```

### 4. Run API Server
```bash
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## System Architecture

### Core Components
- **ResumeScreener**: Main screening engine with BERT + TF-IDF
- **SQLite Database**: Persistent storage for resumes and jobs
- **Streamlit UI**: Interactive web interface
- **FastAPI**: REST API endpoints
- **Mock Data Generator**: Realistic test data

### Scoring Algorithm
1. **BERT Embeddings** (40%): Semantic similarity using SentenceTransformers
2. **TF-IDF Analysis** (30%): Keyword-based matching
3. **Skill Matching** (20%): Jaccard similarity of required vs candidate skills
4. **Experience Score** (10%): Normalized experience level matching

## API Endpoints

### Resumes
- `POST /resumes` - Add new resume
- `GET /resumes` - List all resumes
- `GET /resumes/{id}` - Get specific resume

### Jobs
- `POST /jobs` - Add job description
- `GET /jobs` - List all jobs
- `GET /jobs/{id}` - Get specific job

### Screening
- `POST /screen` - Screen resumes against job
- `GET /screen/results/{job_id}` - Get screening results

### Utilities
- `POST /mock-data` - Generate test data
- `GET /stats` - System statistics
- `GET /health` - Health check

## Example Usage

### Python API
```python
from resume_screener import ResumeScreener

screener = ResumeScreener()

# Add job
job_id = screener.add_job_description(
    "Python Developer", 
    "Looking for Python developer...",
    ["Python", "Django", "AWS"],
    3
)

# Add resume
screener.add_resume("John Doe", "Python developer with 5 years...")

# Screen resumes
results = screener.screen_resumes(job_id)
```

### REST API
```bash
# Add job
curl -X POST "http://localhost:8000/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Developer",
    "content": "Looking for Python developer...",
    "required_skills": ["Python", "Django"],
    "experience_required": 3
  }'

# Screen resumes
curl -X POST "http://localhost:8000/screen" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": 1,
    "weights": {
      "bert": 0.4,
      "tfidf": 0.3,
      "skills": 0.2,
      "experience": 0.1
    }
  }'
```

## Production Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501 8000

CMD ["streamlit", "run", "streamlit_simple.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables
```bash
export DATABASE_PATH=/data/resume_database.db
export MODEL_NAME=all-MiniLM-L6-v2
export API_HOST=0.0.0.0
export API_PORT=8000
```

## Performance Optimization

### Model Caching
- SentenceTransformer models are cached after first load
- Database connections are pooled
- Results are cached for repeated queries

### Scaling
- Use Redis for distributed caching
- Deploy multiple API instances behind load balancer
- Use PostgreSQL for production database

## Monitoring

### Health Checks
- `/health` endpoint for API status
- Database connectivity checks
- Model loading verification

### Metrics
- Response times for screening operations
- Database query performance
- Model inference latency

## Security

### API Security
- Add authentication middleware
- Rate limiting for API endpoints
- Input validation and sanitization

### Data Protection
- Encrypt sensitive resume data
- Implement GDPR compliance
- Audit logging for data access

## Troubleshooting

### Common Issues
1. **Model Download**: First run downloads BERT model (~90MB)
2. **Memory Usage**: BERT models require ~500MB RAM
3. **Database Locks**: Use connection pooling for concurrent access

### Performance Tuning
- Adjust batch sizes for large resume sets
- Use GPU acceleration for BERT if available
- Implement result caching for repeated queries
