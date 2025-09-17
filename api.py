"""
FastAPI REST API for Resume Screening System
Provides RESTful endpoints for programmatic access to screening functionality
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from resume_screener import ResumeScreener
from mock_data import populate_mock_data

# Initialize FastAPI app
app = FastAPI(
    title="Resume Screening API",
    description="AI-powered resume screening system with BERT embeddings and advanced similarity metrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize screener
screener = ResumeScreener()

# Pydantic models
class ResumeCreate(BaseModel):
    name: str
    content: str

class JobCreate(BaseModel):
    title: str
    content: str
    required_skills: Optional[List[str]] = []
    experience_required: Optional[int] = 0

class ScreeningWeights(BaseModel):
    bert: float = 0.4
    tfidf: float = 0.3
    skills: float = 0.2
    experience: float = 0.1

class ScreeningRequest(BaseModel):
    job_id: int
    weights: Optional[ScreeningWeights] = ScreeningWeights()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Resume Screening API",
        "version": "1.0.0",
        "endpoints": {
            "resumes": "/resumes",
            "jobs": "/jobs", 
            "screen": "/screen",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "resume-screening-api"}

# Resume endpoints
@app.post("/resumes", response_model=dict)
async def create_resume(resume: ResumeCreate):
    """Create a new resume"""
    try:
        resume_id = screener.add_resume(resume.name, resume.content)
        return {
            "message": "Resume created successfully",
            "resume_id": resume_id,
            "name": resume.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resumes", response_model=List[dict])
async def get_resumes():
    """Get all resumes"""
    try:
        resumes = screener.get_resumes()
        return resumes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resumes/{resume_id}", response_model=dict)
async def get_resume(resume_id: int):
    """Get a specific resume by ID"""
    try:
        resumes = screener.get_resumes()
        resume = next((r for r in resumes if r['id'] == resume_id), None)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        return resume
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job endpoints
@app.post("/jobs", response_model=dict)
async def create_job(job: JobCreate):
    """Create a new job description"""
    try:
        job_id = screener.add_job_description(
            job.title, 
            job.content, 
            job.required_skills, 
            job.experience_required
        )
        return {
            "message": "Job description created successfully",
            "job_id": job_id,
            "title": job.title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=List[dict])
async def get_jobs():
    """Get all job descriptions"""
    try:
        jobs = screener.get_job_descriptions()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=dict)
async def get_job(job_id: int):
    """Get a specific job description by ID"""
    try:
        jobs = screener.get_job_descriptions()
        job = next((j for j in jobs if j['id'] == job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job description not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Screening endpoints
@app.post("/screen", response_model=List[dict])
async def screen_resumes(request: ScreeningRequest):
    """Screen resumes against a job description"""
    try:
        weights = {
            'bert': request.weights.bert,
            'tfidf': request.weights.tfidf,
            'skills': request.weights.skills,
            'experience': request.weights.experience
        }
        
        results = screener.screen_resumes(request.job_id, weights)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/screen/results/{job_id}", response_model=List[dict])
async def get_screening_results(job_id: int):
    """Get previous screening results for a job"""
    try:
        results = screener.get_screening_results(job_id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.post("/mock-data")
async def generate_mock_data():
    """Generate mock data for testing"""
    try:
        job_ids = populate_mock_data()
        return {
            "message": "Mock data generated successfully",
            "job_ids": job_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        resumes = screener.get_resumes()
        jobs = screener.get_job_descriptions()
        
        # Calculate statistics
        total_resumes = len(resumes)
        total_jobs = len(jobs)
        avg_experience = sum(r['experience_years'] for r in resumes) / total_resumes if resumes else 0
        
        # Count unique skills
        all_skills = []
        for resume in resumes:
            if resume['skills']:
                all_skills.extend(resume['skills'])
        unique_skills = len(set(all_skills))
        
        return {
            "total_resumes": total_resumes,
            "total_jobs": total_jobs,
            "average_experience": round(avg_experience, 1),
            "unique_skills": unique_skills,
            "most_common_skills": list(set(all_skills))[:10] if all_skills else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Upload a resume file (text content)"""
    try:
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.filename.endswith('.txt'):
            resume_content = content.decode('utf-8')
        else:
            # For now, only support text files
            # In the future, add PDF/Word parsing
            raise HTTPException(
                status_code=400, 
                detail="Only .txt files are currently supported"
            )
        
        # Create resume
        resume_id = screener.add_resume(name, resume_content)
        
        return {
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "filename": file.filename,
            "name": name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
