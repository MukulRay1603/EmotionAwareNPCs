# Backend Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation Steps

### 1. Create Virtual Environment
```bash
cd backend
python3 -m venv venv
```

### 2. Activate Virtual Environment
**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
python main.py
```

The server will start at `http://localhost:8000`

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Latest Emotion
```bash
curl http://localhost:8000/infer
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation

## Configuration
Create a `.env` file in the backend directory for configuration:
```
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

## Troubleshooting
- If port 8000 is in use, change the port in `main.py`
- Ensure all dependencies are installed correctly
- Check Python version compatibility

