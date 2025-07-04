# Advanced Image Background Remover API

![Background Removal Example](https://i.postimg.cc/W4XmgVCK/result2.png) <!-- Replace with actual image -->

A high-performance background removal API built with Python, FastAPI, and state-of-the-art computer vision models. Easily integrate professional image processing capabilities into your applications.

## Key Features
- 🚀 Real-time background removal with GPU acceleration
- ⚡️ Asynchronous processing for large images
- 🔒 Secure token-based authentication
- 📊 Health monitoring and model information endpoints
- 🧠 Automatic model caching for faster subsequent requests
- 🌐 Cross-platform support (Windows, Linux, macOS)

## Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3-pip python3-venv redis-server
sudo apt install libgl1-mesa-glx libglib2.0-0  # OpenCV dependencies
```

### macOS (Homebrew)
```bash
brew install redis
brew install python3
```

### Windows
1. Install Python from [python.org](https://python.org)
2. Install Redis via WSL or use [Redis Cloud](https://redis.com/try-free/)

## Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
# Start Redis server (in a separate terminal)
redis-server

# Run the API in development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** Models will be automatically downloaded on first use (approx. 500MB). Subsequent runs will use cached models.

## API Usage

### Authentication
All endpoints require an access token:
```http
Authorization: Bearer your-access-token-here
```

### Endpoints

1. **Basic Background Removal**  
   `POST /remove-background`
   ```bash
   curl -X POST http://localhost:8000/remove-background \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@input.jpg" \
     -F "enhance=true" \
     --output result.png
   ```

2. **Asynchronous Processing**  
   `POST /remove-background-async`
   ```bash
   # Submit task
   curl -X POST http://localhost:8000/remove-background-async \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@large_image.jpg"
   
   # Check status
   curl -X GET "http://localhost:8000/task-status/TASK_ID" \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

3. **Model Information**  
   `GET /models`
   ```bash
   curl -X GET "http://localhost:8000/models" \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

4. **Health Check**  
   `GET /health`
   ```bash
   curl -X GET "http://localhost:8000/health" \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

## Python Client Example

```python
import requests

url = "http://localhost:8000/remove-background"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

with open("input.jpg", "rb") as f:
    response = requests.post(
        url, 
        headers=headers, 
        files={"file": f}, 
        data={"enhance": True}
    )

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Background removed successfully!")
else:
    print(f"Error {response.status_code}: {response.text}")
```

## Configuration

Configure environment variables in `.env` file:

```env
# File handling settings
MAX_FILE_SIZE=10  # Maximum file size in MB
CACHE_TTL=3600    # Cache time-to-live in seconds

# Processing configuration
MAX_WORKERS=4                    # Number of CPU threads for processing
BG_REMOVAL_MODEL=u2net          # Background removal model (u2net, u2netp, silueta)
TEMP_DIR=/tmp/bg_removal       

# Performance tuning
UVICORN_WORKERS=1               # Number of worker processes
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to transform your images?** Clone the repository and start removing backgrounds in minutes!