#!/usr/bin/env python3
"""
Setup script for Emotion-Aware NPCs project
Sets up the entire development environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Main setup class for the project"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        if self.python_version < (3, 8):
            logger.error("Python 3.8+ is required. Current version: {}.{}.{}".format(
                self.python_version.major, self.python_version.minor, self.python_version.micro
            ))
            return False
        
        logger.info(f"Python version OK: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        return True
    
    def setup_backend(self):
        """Setup backend environment"""
        logger.info("Setting up backend...")
        
        backend_dir = self.project_root / "backend"
        os.chdir(backend_dir)
        
        try:
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            logger.info("Created virtual environment")
            
            # Get activation script path
            if platform.system() == "Windows":
                activate_script = backend_dir / "venv" / "Scripts" / "activate.bat"
                pip_path = backend_dir / "venv" / "Scripts" / "pip"
            else:
                activate_script = backend_dir / "venv" / "bin" / "activate"
                pip_path = backend_dir / "venv" / "bin" / "pip"
            
            # Install requirements
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            logger.info("Installed backend requirements")
            
            # Test backend
            logger.info("Testing backend...")
            subprocess.run([str(pip_path), "install", "pytest"], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Backend setup failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def setup_cv(self):
        """Setup CV module"""
        logger.info("Setting up CV module...")
        
        cv_dir = self.project_root / "cv"
        os.chdir(cv_dir)
        
        try:
            # Run CV setup script
            subprocess.run([sys.executable, "setup_cv.py"], check=True)
            logger.info("CV module setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"CV setup failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def setup_unity_instructions(self):
        """Display Unity setup instructions"""
        logger.info("Unity setup instructions:")
        print("\n" + "="*60)
        print("UNITY SETUP INSTRUCTIONS")
        print("="*60)
        print("1. Install Unity Hub from https://unity.com/download")
        print("2. Install Unity 2022.3 LTS or higher")
        print("3. Open Unity Hub and click 'Add'")
        print("4. Navigate to the 'unity' folder in this project")
        print("5. Select the project and click 'Open'")
        print("6. Import required packages:")
        print("   - TextMeshPro (Window > TextMeshPro > Import TMP Essential Resources)")
        print("   - Newtonsoft.Json (Window > Package Manager > Search 'Newtonsoft Json')")
        print("7. Open the main scene and press Play")
        print("="*60)
    
    def create_launch_scripts(self):
        """Create launch scripts for different platforms"""
        logger.info("Creating launch scripts...")
        
        # Backend launch script
        if platform.system() == "Windows":
            backend_script = """@echo off
cd backend
call venv\\Scripts\\activate
python main.py
pause
"""
            with open("start_backend.bat", "w") as f:
                f.write(backend_script)
        else:
            backend_script = """#!/bin/bash
cd backend
source venv/bin/activate
python main.py
"""
            with open("start_backend.sh", "w") as f:
                f.write(backend_script)
            os.chmod("start_backend.sh", 0o755)
        
        # CV inference script
        if platform.system() == "Windows":
            cv_script = """@echo off
cd cv
python inference/webcam_inference.py
pause
"""
            with open("start_cv.bat", "w") as f:
                f.write(cv_script)
        else:
            cv_script = """#!/bin/bash
cd cv
python inference/webcam_inference.py
"""
            with open("start_cv.sh", "w") as f:
                f.write(cv_script)
            os.chmod("start_cv.sh", 0o755)
        
        logger.info("Created launch scripts")
    
    def create_docker_setup(self):
        """Create Docker setup for easy deployment"""
        logger.info("Creating Docker setup...")
        
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY cv/ ./cv/

# Expose port
EXPOSE 8000

# Set working directory
WORKDIR /app/backend

# Run the application
CMD ["python", "main.py"]
"""
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Docker compose
        docker_compose_content = """version: '3.8'

services:
  emotion-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./cv/output:/app/cv/output
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
"""
        
        with open("docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        
        logger.info("Created Docker configuration")
    
    def run_tests(self):
        """Run basic tests to verify setup"""
        logger.info("Running tests...")
        
        try:
            # Test backend
            backend_dir = self.project_root / "backend"
            os.chdir(backend_dir)
            
            if platform.system() == "Windows":
                python_path = backend_dir / "venv" / "Scripts" / "python"
            else:
                python_path = backend_dir / "venv" / "bin" / "python"
            
            # Test import
            result = subprocess.run([str(python_path), "-c", "import main; print('Backend import OK')"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Backend test passed")
            else:
                logger.error(f"Backend test failed: {result.stderr}")
            
            # Test CV module
            os.chdir(self.project_root / "cv")
            result = subprocess.run([sys.executable, "test_cv.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("CV module test passed")
            else:
                logger.error(f"CV module test failed: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def display_next_steps(self):
        """Display next steps for the user"""
        print("\n" + "="*60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Start the backend server:")
        if platform.system() == "Windows":
            print("   .\\start_backend.bat")
        else:
            print("   ./start_backend.sh")
        
        print("\n2. Test the API:")
        print("   curl http://localhost:8000/health")
        print("   curl http://localhost:8000/infer")
        
        print("\n3. Start CV inference (optional):")
        if platform.system() == "Windows":
            print("   .\\start_cv.bat")
        else:
            print("   ./start_cv.sh")
        
        print("\n4. Setup Unity (see instructions above)")
        
        print("\n5. For Docker deployment:")
        print("   docker-compose up --build")
        
        print("\n6. View API documentation:")
        print("   http://localhost:8000/docs")
        
        print("\n" + "="*60)
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("Starting Emotion-Aware NPCs setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Setup backend
        if not self.setup_backend():
            logger.error("Backend setup failed")
            return False
        
        # Setup CV module
        if not self.setup_cv():
            logger.error("CV setup failed")
            return False
        
        # Create launch scripts
        self.create_launch_scripts()
        
        # Create Docker setup
        self.create_docker_setup()
        
        # Run tests
        if not self.run_tests():
            logger.warning("Some tests failed, but setup may still work")
        
        # Display Unity instructions
        self.setup_unity_instructions()
        
        # Display next steps
        self.display_next_steps()
        
        logger.info("Setup completed successfully!")
        return True

def main():
    """Main function"""
    setup = ProjectSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
