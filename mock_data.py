"""
Mock data generator for Resume Screening System
Creates realistic sample resumes and job descriptions for testing
"""

import json
import random
from resume_screener import ResumeScreener

class MockDataGenerator:
    def __init__(self):
        self.tech_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust", "Swift", "Kotlin",
            "React", "Angular", "Vue", "Django", "Flask", "Spring", "Express", "Laravel", "FastAPI",
            "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "SQLite", "Cassandra",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "Linux", "Terraform",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science",
            "TensorFlow", "PyTorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
            "Hadoop", "Spark", "Kafka", "Airflow", "Tableau", "Power BI", "Jupyter", "R"
        ]
        
        self.soft_skills = [
            "problem-solving", "communication", "teamwork", "leadership", "analytical thinking",
            "project management", "agile methodology", "scrum", "critical thinking", "creativity"
        ]
        
        self.education_levels = [
            "Bachelor's in Computer Science",
            "Master's in Data Science", 
            "PhD in Machine Learning",
            "Bachelor's in Software Engineering",
            "Master's in Computer Science",
            "Bachelor's in Information Technology",
            "Master's in Artificial Intelligence",
            "Bachelor's in Mathematics",
            "Master's in Statistics"
        ]
        
        self.companies = [
            "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Tesla", "Uber",
            "Airbnb", "Spotify", "Adobe", "Salesforce", "Oracle", "IBM", "Intel", "NVIDIA"
        ]
        
    def generate_resume(self, name: str, experience_years: int = None) -> str:
        """Generate a realistic resume"""
        if experience_years is None:
            experience_years = random.randint(1, 10)
            
        # Select random skills
        num_tech_skills = random.randint(5, 12)
        selected_tech_skills = random.sample(self.tech_skills, num_tech_skills)
        
        num_soft_skills = random.randint(3, 6)
        selected_soft_skills = random.sample(self.soft_skills, num_soft_skills)
        
        education = random.choice(self.education_levels)
        companies_worked = random.sample(self.companies, min(experience_years, 3))
        
        resume = f"""
{name}
Software Engineer | {experience_years} years of experience

EDUCATION:
{education}

TECHNICAL SKILLS:
{', '.join(selected_tech_skills)}

PROFESSIONAL EXPERIENCE:
"""
        
        for i, company in enumerate(companies_worked):
            years_at_company = max(1, experience_years // len(companies_worked))
            resume += f"""
{company} - Senior Software Engineer ({years_at_company} years)
• Developed and maintained scalable applications using {random.choice(selected_tech_skills[:3])}
• Collaborated with cross-functional teams to deliver high-quality software solutions
• Implemented {random.choice(['microservices architecture', 'CI/CD pipelines', 'automated testing', 'cloud infrastructure'])}
• Mentored junior developers and conducted code reviews
"""

        resume += f"""
SOFT SKILLS:
{', '.join(selected_soft_skills)}

ACHIEVEMENTS:
• Led development of {random.choice(['customer-facing application', 'internal tool', 'data pipeline', 'ML model'])} that improved {random.choice(['performance by 40%', 'user engagement by 25%', 'processing speed by 60%', 'accuracy by 30%'])}
• Contributed to open-source projects and technical blog posts
• Certified in {random.choice(['AWS', 'Azure', 'GCP', 'Kubernetes', 'Scrum Master'])}
"""
        
        return resume.strip()
    
    def generate_job_description(self, title: str, required_experience: int = None) -> tuple:
        """Generate a realistic job description"""
        if required_experience is None:
            required_experience = random.randint(2, 8)
            
        # Select required skills based on job title
        if "Data Scientist" in title or "ML" in title:
            core_skills = ["Python", "pandas", "scikit-learn", "TensorFlow", "PyTorch", "SQL"]
            additional_skills = ["AWS", "Docker", "Jupyter", "R", "Tableau", "Spark"]
        elif "Frontend" in title or "React" in title:
            core_skills = ["JavaScript", "React", "HTML", "CSS", "TypeScript"]
            additional_skills = ["Angular", "Vue", "Node.js", "Webpack", "Jest"]
        elif "Backend" in title or "API" in title:
            core_skills = ["Python", "Django", "Flask", "PostgreSQL", "Redis"]
            additional_skills = ["AWS", "Docker", "Kubernetes", "MongoDB", "Elasticsearch"]
        elif "DevOps" in title or "Cloud" in title:
            core_skills = ["AWS", "Docker", "Kubernetes", "Jenkins", "Terraform"]
            additional_skills = ["Python", "Linux", "Ansible", "Prometheus", "Grafana"]
        else:  # Full Stack or General
            core_skills = ["Python", "JavaScript", "React", "Django", "PostgreSQL"]
            additional_skills = ["AWS", "Docker", "Redis", "Git", "Linux"]
        
        # Combine skills
        required_skills = core_skills + random.sample(additional_skills, min(3, len(additional_skills)))
        
        job_description = f"""
{title}

We are seeking a talented {title} with {required_experience}+ years of experience to join our growing team.

RESPONSIBILITIES:
• Design and develop scalable software solutions using modern technologies
• Collaborate with product managers and designers to deliver exceptional user experiences  
• Write clean, maintainable, and well-tested code
• Participate in code reviews and contribute to technical discussions
• Mentor junior team members and share knowledge across the organization
• Stay up-to-date with industry trends and best practices

REQUIRED QUALIFICATIONS:
• {required_experience}+ years of professional software development experience
• Strong proficiency in {', '.join(core_skills[:3])}
• Experience with {', '.join(core_skills[3:])} 
• Solid understanding of software engineering principles and design patterns
• Experience with version control systems (Git) and agile development methodologies
• Strong problem-solving skills and attention to detail
• Excellent communication and collaboration skills

PREFERRED QUALIFICATIONS:
• Experience with {', '.join(additional_skills[:3])}
• Knowledge of {random.choice(['microservices architecture', 'cloud platforms', 'containerization', 'CI/CD pipelines'])}
• Contributions to open-source projects
• Bachelor's degree in Computer Science or related field

WHAT WE OFFER:
• Competitive salary and equity package
• Comprehensive health, dental, and vision insurance
• Flexible work arrangements and unlimited PTO
• Professional development opportunities and conference attendance
• State-of-the-art equipment and modern office environment
"""
        
        return job_description.strip(), required_skills, required_experience

def populate_mock_data():
    """Populate the database with mock data"""
    screener = ResumeScreener()
    generator = MockDataGenerator()
    
    # Generate job descriptions
    job_titles = [
        "Senior Python Developer",
        "Data Scientist - Machine Learning",
        "Frontend React Developer", 
        "Backend API Engineer",
        "Full Stack Developer",
        "DevOps Engineer",
        "ML Engineer",
        "Software Architect"
    ]
    
    job_ids = []
    print("Creating job descriptions...")
    for title in job_titles:
        job_desc, skills, exp = generator.generate_job_description(title)
        job_id = screener.add_job_description(title, job_desc, skills, exp)
        job_ids.append(job_id)
        print(f"✓ Created: {title}")
    
    # Generate resumes
    candidate_names = [
        "Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Emma Brown",
        "Frank Miller", "Grace Lee", "Henry Taylor", "Ivy Chen", "Jack Anderson",
        "Kate Thompson", "Liam Garcia", "Maya Patel", "Noah Rodriguez", "Olivia Martinez",
        "Paul Jackson", "Quinn White", "Rachel Green", "Sam Kim", "Tina Liu"
    ]
    
    print("\nCreating candidate resumes...")
    for name in candidate_names:
        experience = random.randint(1, 10)
        resume_content = generator.generate_resume(name, experience)
        screener.add_resume(name, resume_content)
        print(f"✓ Created resume for: {name} ({experience} years exp)")
    
    print(f"\n✅ Mock data generation complete!")
    print(f"Created {len(job_titles)} job descriptions and {len(candidate_names)} resumes")
    
    return job_ids

if __name__ == "__main__":
    populate_mock_data()
