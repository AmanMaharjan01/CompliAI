"""
Initialize database schema
"""

import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/policydb"
)

def init_database():
    """Create database tables"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Create users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                role VARCHAR(20) CHECK (role IN ('admin', 'employee')) DEFAULT 'employee',
                department VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create documents table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                file_type VARCHAR(20),
                department VARCHAR(100),
                policy_type VARCHAR(100),
                effective_date DATE,
                description TEXT,
                uploaded_by UUID REFERENCES users(id),
                uploaded_at TIMESTAMP DEFAULT NOW(),
                indexed BOOLEAN DEFAULT FALSE,
                num_chunks INTEGER DEFAULT 0
            )
        """))
        
        # Create query logs table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id),
                question TEXT NOT NULL,
                answer TEXT,
                confidence_score FLOAT,
                retrieval_time_ms INTEGER,
                generation_time_ms INTEGER,
                total_time_ms INTEGER,
                num_sources INTEGER,
                helpful BOOLEAN,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create analytics table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analytics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                date DATE NOT NULL,
                total_queries INTEGER DEFAULT 0,
                avg_response_time_ms FLOAT,
                avg_confidence FLOAT,
                top_questions JSONB,
                department_breakdown JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        conn.commit()
        print("Database initialized successfully!")


if __name__ == "__main__":
    init_database()
