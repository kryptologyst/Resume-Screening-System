"""
Simplified Resume Screening System - Streamlit Web Interface
Works with basic dependencies for immediate deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
from resume_screener import ResumeScreener
from mock_data import MockDataGenerator, populate_mock_data
import json
import sqlite3
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: black;
    }
    
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
        color: black;
    }
    
    .job-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_screener():
    """Initialize and cache the resume screener"""
    return ResumeScreener()

@st.cache_data
def load_data():
    """Load data from database"""
    screener = get_screener()
    resumes = screener.get_resumes()
    jobs = screener.get_job_descriptions()
    return resumes, jobs

def display_screening_results(results):
    """Display screening results with basic visualizations"""
    if not results:
        st.warning("No screening results found.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Top candidates summary
    st.subheader("🏆 Top Candidates")
    
    cols = st.columns(3)
    for i, result in enumerate(results[:3]):
        with cols[i]:
            st.markdown(f"""
            <div class="candidate-card">
                <h4>#{result['ranking']} {result['resume_name']}</h4>
                <p><strong>Final Score:</strong> {result['final_score']:.3f}</p>
                <p><strong>BERT:</strong> {result['bert_score']:.3f} | 
                   <strong>TF-IDF:</strong> {result['tfidf_score']:.3f}</p>
                <p><strong>Skills:</strong> {result['skill_score']:.3f} | 
                   <strong>Experience:</strong> {result['experience_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed results table
    st.subheader("📊 Detailed Results")
    
    # Create interactive table
    display_df = df[['ranking', 'resume_name', 'final_score', 'bert_score', 
                     'tfidf_score', 'skill_score', 'experience_score']].copy()
    display_df.columns = ['Rank', 'Candidate', 'Final Score', 'BERT Score', 
                          'TF-IDF Score', 'Skill Match', 'Experience']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Simple bar chart
    st.subheader("📈 Score Comparison")
    chart_data = df[['resume_name', 'final_score']].set_index('resume_name')
    st.bar_chart(chart_data)

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">📄 Resume Screening System</h1>', unsafe_allow_html=True)
    
    # Initialize screener
    screener = get_screener()
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Data management
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("🔄 Generate Mock Data"):
        with st.spinner("Generating mock data..."):
            populate_mock_data()
        st.sidebar.success("Mock data generated!")
        st.experimental_rerun()
    
    # Load data
    resumes, jobs = load_data()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard", 
        "📋 Screen Resumes", 
        "📄 Manage Resumes", 
        "💼 Manage Jobs"
    ])
    
    with tab1:
        st.header("Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(resumes)}</h3>
                <p>Total Resumes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(jobs)}</h3>
                <p>Job Descriptions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_exp = np.mean([r['experience_years'] for r in resumes]) if resumes else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_exp:.1f}</h3>
                <p>Avg Experience (Years)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Count unique skills
            all_skills = []
            for resume in resumes:
                if resume['skills']:
                    all_skills.extend(resume['skills'])
            unique_skills = len(set(all_skills))
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_skills}</h3>
                <p>Unique Skills</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        if resumes:
            recent_resumes = sorted(resumes, key=lambda x: x['created_at'], reverse=True)[:5]
            for resume in recent_resumes:
                st.markdown(f"""
                <div class="candidate-card">
                    <strong>{resume['name']}</strong> - {resume['experience_years']} years experience
                    <br><small>Added: {resume['created_at']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Skills distribution
        if resumes:
            st.subheader("🔧 Top Skills")
            skill_counts = {}
            for resume in resumes:
                if resume['skills']:
                    for skill in resume['skills']:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            if skill_counts:
                top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
                st.bar_chart(skills_df.set_index('Skill'))
    
    with tab2:
        st.header("Screen Resumes Against Job")
        
        if not jobs:
            st.warning("No job descriptions available. Please add some jobs first or generate mock data.")
            return
        
        # Job selection
        job_options = {f"{job['title']} (ID: {job['id']})": job['id'] for job in jobs}
        selected_job = st.selectbox("Select Job Description", options=list(job_options.keys()))
        
        if selected_job:
            job_id = job_options[selected_job]
            
            # Display selected job details
            selected_job_data = next(job for job in jobs if job['id'] == job_id)
            
            st.markdown(f"""
            <div class="job-card">
                <h4>{selected_job_data['title']}</h4>
                <p><strong>Required Experience:</strong> {selected_job_data['experience_required']} years</p>
                <p><strong>Required Skills:</strong> {', '.join(selected_job_data['required_skills'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Screening parameters
            st.subheader("⚙️ Screening Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                bert_weight = st.slider("BERT Weight", 0.0, 1.0, 0.4, 0.1)
                tfidf_weight = st.slider("TF-IDF Weight", 0.0, 1.0, 0.3, 0.1)
            
            with col2:
                skill_weight = st.slider("Skill Match Weight", 0.0, 1.0, 0.2, 0.1)
                exp_weight = st.slider("Experience Weight", 0.0, 1.0, 0.1, 0.1)
            
            # Normalize weights
            total_weight = bert_weight + tfidf_weight + skill_weight + exp_weight
            if total_weight > 0:
                weights = {
                    'bert': bert_weight / total_weight,
                    'tfidf': tfidf_weight / total_weight,
                    'skills': skill_weight / total_weight,
                    'experience': exp_weight / total_weight
                }
            else:
                weights = {'bert': 0.4, 'tfidf': 0.3, 'skills': 0.2, 'experience': 0.1}
            
            # Screen resumes button
            if st.button("🔍 Screen Resumes", type="primary"):
                with st.spinner("Screening resumes..."):
                    results = screener.screen_resumes(job_id, weights)
                
                if results:
                    st.success(f"Screened {len(results)} resumes successfully!")
                    display_screening_results(results)
                else:
                    st.warning("No resumes found to screen.")
    
    with tab3:
        st.header("Manage Resumes")
        
        # Add new resume
        st.subheader("➕ Add New Resume")
        
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Candidate Name")
        with col2:
            new_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
        
        new_content = st.text_area("Resume Content", height=200, 
                                   placeholder="Enter the full resume content here...")
        
        if st.button("Add Resume"):
            if new_name and new_content:
                resume_id = screener.add_resume(new_name, new_content)
                st.success(f"Resume added successfully! ID: {resume_id}")
                st.experimental_rerun()
            else:
                st.error("Please fill in all required fields.")
        
        # Display existing resumes
        st.subheader("📋 Existing Resumes")
        
        if resumes:
            for resume in resumes:
                with st.expander(f"{resume['name']} - {resume['experience_years']} years"):
                    st.write(f"**Skills:** {', '.join(resume['skills']) if resume['skills'] else 'None detected'}")
                    st.write(f"**Added:** {resume['created_at']}")
                    st.text_area("Content", value=resume['content'], height=150, disabled=True, key=f"resume_{resume['id']}")
        else:
            st.info("No resumes found. Add some resumes or generate mock data.")
    
    with tab4:
        st.header("Manage Job Descriptions")
        
        # Add new job
        st.subheader("➕ Add New Job Description")
        
        col1, col2 = st.columns(2)
        with col1:
            new_job_title = st.text_input("Job Title")
        with col2:
            new_job_experience = st.number_input("Required Experience (Years)", min_value=0, max_value=20, value=3)
        
        new_job_skills = st.text_input("Required Skills (comma-separated)", 
                                       placeholder="Python, Django, AWS, Docker")
        new_job_content = st.text_area("Job Description", height=200,
                                       placeholder="Enter the full job description here...")
        
        if st.button("Add Job Description"):
            if new_job_title and new_job_content:
                skills_list = [skill.strip() for skill in new_job_skills.split(',')] if new_job_skills else []
                job_id = screener.add_job_description(new_job_title, new_job_content, skills_list, new_job_experience)
                st.success(f"Job description added successfully! ID: {job_id}")
                st.experimental_rerun()
            else:
                st.error("Please fill in all required fields.")
        
        # Display existing jobs
        st.subheader("💼 Existing Job Descriptions")
        
        if jobs:
            for job in jobs:
                with st.expander(f"{job['title']} - {job['experience_required']} years required"):
                    st.write(f"**Required Skills:** {', '.join(job['required_skills']) if job['required_skills'] else 'None specified'}")
                    st.write(f"**Added:** {job['created_at']}")
                    st.text_area("Content", value=job['content'], height=150, disabled=True, key=f"job_{job['id']}")
        else:
            st.info("No job descriptions found. Add some jobs or generate mock data.")

if __name__ == "__main__":
    main()
