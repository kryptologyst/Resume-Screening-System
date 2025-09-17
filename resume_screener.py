"""
Modern Resume Screening System
Uses state-of-the-art NLP techniques including BERT embeddings and advanced similarity metrics
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import re
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScreener:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Resume Screener with modern NLP models
        
        Args:
            model_name: SentenceTransformer model name for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.db_path = 'resume_database.db'
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing resumes and job descriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                skills TEXT,
                experience_years INTEGER,
                education TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                required_skills TEXT,
                experience_required INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS screening_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                resume_id INTEGER,
                similarity_score REAL,
                bert_score REAL,
                tfidf_score REAL,
                final_score REAL,
                ranking INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using pattern matching"""
        # Common technical skills patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|SQLite)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|Linux)\b',
            r'\b(?:Machine Learning|Deep Learning|NLP|Computer Vision|Data Science)\b',
            r'\b(?:TensorFlow|PyTorch|scikit-learn|pandas|numpy|matplotlib)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([match.lower() for match in matches])
        
        return list(set(skills))
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from resume text"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*:\s*(\d+)\+?\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return max([int(match) for match in matches])
        
        return 0
    
    def add_resume(self, name: str, content: str) -> int:
        """Add a resume to the database"""
        skills = self.extract_skills(content)
        experience_years = self.extract_experience_years(content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resumes (name, content, skills, experience_years)
            VALUES (?, ?, ?, ?)
        ''', (name, content, json.dumps(skills), experience_years))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added resume for {name} with ID {resume_id}")
        return resume_id
    
    def add_job_description(self, title: str, content: str, required_skills: List[str] = None, experience_required: int = 0) -> int:
        """Add a job description to the database"""
        if required_skills is None:
            required_skills = self.extract_skills(content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO job_descriptions (title, content, required_skills, experience_required)
            VALUES (?, ?, ?, ?)
        ''', (title, content, json.dumps(required_skills), experience_required))
        
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added job description '{title}' with ID {job_id}")
        return job_id
    
    def get_resumes(self) -> List[Dict]:
        """Retrieve all resumes from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM resumes')
        rows = cursor.fetchall()
        
        resumes = []
        for row in rows:
            resumes.append({
                'id': row[0],
                'name': row[1],
                'content': row[2],
                'skills': json.loads(row[3]) if row[3] else [],
                'experience_years': row[4],
                'education': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return resumes
    
    def get_job_descriptions(self) -> List[Dict]:
        """Retrieve all job descriptions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM job_descriptions')
        rows = cursor.fetchall()
        
        jobs = []
        for row in rows:
            jobs.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'required_skills': json.loads(row[3]) if row[3] else [],
                'experience_required': row[4],
                'created_at': row[5]
            })
        
        conn.close()
        return jobs
    
    def compute_bert_similarity(self, job_description: str, resumes: List[str]) -> np.ndarray:
        """Compute BERT-based semantic similarity"""
        # Generate embeddings
        job_embedding = self.model.encode([job_description])
        resume_embeddings = self.model.encode(resumes)
        
        # Compute cosine similarity
        similarities = cosine_similarity(job_embedding, resume_embeddings).flatten()
        return similarities
    
    def compute_tfidf_similarity(self, job_description: str, resumes: List[str]) -> np.ndarray:
        """Compute TF-IDF based similarity"""
        documents = [job_description] + resumes
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(job_vector, resume_vectors).flatten()
        return similarities
    
    def compute_skill_match_score(self, job_skills: List[str], resume_skills: List[str]) -> float:
        """Compute skill matching score"""
        if not job_skills or not resume_skills:
            return 0.0
        
        job_skills_set = set([skill.lower() for skill in job_skills])
        resume_skills_set = set([skill.lower() for skill in resume_skills])
        
        intersection = job_skills_set.intersection(resume_skills_set)
        union = job_skills_set.union(resume_skills_set)
        
        # Jaccard similarity
        jaccard_score = len(intersection) / len(union) if union else 0
        
        # Bonus for having more required skills
        required_skills_match = len(intersection) / len(job_skills_set) if job_skills_set else 0
        
        return (jaccard_score + required_skills_match) / 2
    
    def screen_resumes(self, job_id: int, weights: Dict[str, float] = None) -> List[Dict]:
        """
        Screen resumes against a job description using multiple similarity metrics
        
        Args:
            job_id: ID of the job description
            weights: Weights for different similarity metrics
        """
        if weights is None:
            weights = {
                'bert': 0.4,
                'tfidf': 0.3,
                'skills': 0.2,
                'experience': 0.1
            }
        
        # Get job description
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM job_descriptions WHERE id = ?', (job_id,))
        job_row = cursor.fetchone()
        
        if not job_row:
            raise ValueError(f"Job description with ID {job_id} not found")
        
        job_data = {
            'id': job_row[0],
            'title': job_row[1],
            'content': job_row[2],
            'required_skills': json.loads(job_row[3]) if job_row[3] else [],
            'experience_required': job_row[4]
        }
        
        # Get all resumes
        resumes = self.get_resumes()
        
        if not resumes:
            logger.warning("No resumes found in database")
            return []
        
        # Extract resume contents and data
        resume_contents = [resume['content'] for resume in resumes]
        
        # Compute similarities
        bert_scores = self.compute_bert_similarity(job_data['content'], resume_contents)
        tfidf_scores = self.compute_tfidf_similarity(job_data['content'], resume_contents)
        
        results = []
        
        for i, resume in enumerate(resumes):
            # Skill matching score
            skill_score = self.compute_skill_match_score(
                job_data['required_skills'], 
                resume['skills']
            )
            
            # Experience score (normalized)
            exp_score = min(resume['experience_years'] / max(job_data['experience_required'], 1), 1.0)
            
            # Compute final weighted score
            final_score = (
                weights['bert'] * bert_scores[i] +
                weights['tfidf'] * tfidf_scores[i] +
                weights['skills'] * skill_score +
                weights['experience'] * exp_score
            )
            
            results.append({
                'resume_id': resume['id'],
                'resume_name': resume['name'],
                'bert_score': float(bert_scores[i]),
                'tfidf_score': float(tfidf_scores[i]),
                'skill_score': skill_score,
                'experience_score': exp_score,
                'final_score': final_score,
                'resume_content': resume['content'][:200] + "..." if len(resume['content']) > 200 else resume['content']
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results):
            result['ranking'] = i + 1
        
        # Save results to database
        self._save_screening_results(job_id, results)
        
        conn.close()
        return results
    
    def _save_screening_results(self, job_id: int, results: List[Dict]):
        """Save screening results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear previous results for this job
        cursor.execute('DELETE FROM screening_results WHERE job_id = ?', (job_id,))
        
        # Insert new results
        for result in results:
            cursor.execute('''
                INSERT INTO screening_results 
                (job_id, resume_id, similarity_score, bert_score, tfidf_score, final_score, ranking)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id,
                result['resume_id'],
                result['final_score'],
                result['bert_score'],
                result['tfidf_score'],
                result['final_score'],
                result['ranking']
            ))
        
        conn.commit()
        conn.close()
    
    def get_screening_results(self, job_id: int) -> List[Dict]:
        """Retrieve screening results for a job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sr.*, r.name, r.content, j.title
            FROM screening_results sr
            JOIN resumes r ON sr.resume_id = r.id
            JOIN job_descriptions j ON sr.job_id = j.id
            WHERE sr.job_id = ?
            ORDER BY sr.ranking
        ''', (job_id,))
        
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'job_id': row[1],
                'resume_id': row[2],
                'similarity_score': row[3],
                'bert_score': row[4],
                'tfidf_score': row[5],
                'final_score': row[6],
                'ranking': row[7],
                'created_at': row[8],
                'resume_name': row[9],
                'resume_content': row[10],
                'job_title': row[11]
            })
        
        conn.close()
        return results

if __name__ == "__main__":
    # Example usage
    screener = ResumeScreener()
    
    # Add sample job description
    job_content = """
    We are looking for a Senior Python Developer with 3+ years of experience in machine learning and web development.
    Required skills: Python, Django, scikit-learn, pandas, AWS, Docker.
    Experience with deep learning frameworks like TensorFlow or PyTorch is a plus.
    Strong problem-solving skills and ability to work in agile environments.
    """
    
    job_id = screener.add_job_description(
        "Senior Python Developer", 
        job_content,
        ["Python", "Django", "scikit-learn", "pandas", "AWS", "Docker"],
        3
    )
    
    # Add sample resumes
    resumes_data = [
        ("John Smith", "Experienced Python developer with 5 years in Django and machine learning. Proficient in scikit-learn, pandas, and AWS deployment. Built scalable web applications and ML pipelines."),
        ("Jane Doe", "Data scientist with 4 years experience in Python, pandas, and deep learning. Worked extensively with TensorFlow and PyTorch on computer vision projects. Strong background in statistics."),
        ("Mike Johnson", "Frontend developer with 3 years in React and JavaScript. Some experience with Python and basic machine learning concepts. Looking to transition to full-stack development."),
        ("Sarah Wilson", "Software engineer with 2 years in Java and Spring Boot. Recently completed machine learning bootcamp with focus on Python and scikit-learn. Eager to apply ML skills in production."),
        ("David Brown", "Senior Python developer with 6 years experience. Expert in Django, Flask, and cloud deployment on AWS. Built multiple ML-powered applications using scikit-learn and pandas.")
    ]
    
    for name, content in resumes_data:
        screener.add_resume(name, content)
    
    # Screen resumes
    results = screener.screen_resumes(job_id)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESUME SCREENING RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nRank #{result['ranking']}: {result['resume_name']}")
        print(f"Final Score: {result['final_score']:.3f}")
        print(f"BERT Score: {result['bert_score']:.3f} | TF-IDF Score: {result['tfidf_score']:.3f}")
        print(f"Skill Match: {result['skill_score']:.3f} | Experience: {result['experience_score']:.3f}")
        print(f"Resume Preview: {result['resume_content']}")
        print("-" * 80)
