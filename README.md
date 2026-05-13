# ML Flask Server Documentation

## Overview
Separate Flask server for ML document extraction processing as a microservice.

## Architecture
```
Django Backend (Port 8000)
    ↓ HTTP Request
Flask ML Server (Port 5001)
    ↓ ML Processing
Response with extracted data
```

## Features
- Independent Flask server for ML processing
- Real ML model inference (LayoutLMv3 for document extraction)
- RESTful API endpoints
- Health checks and monitoring
- Batch processing support
- Error handling and logging
- Support for hospital bills, pharmacy bills, KYC documents (Aadhaar, PAN)

## API Endpoints

### Health Check
```
GET http://localhost:5001/health
```

### Single Document Extraction
```
POST http://localhost:5001/extract
Content-Type: multipart/form-data

Form Data:
- file: Document file (image/PDF)
- document_type: hospital_bill | pharmacy_bill | aadhaar | pan
```

### Batch Document Extraction
```
POST http://localhost:5001/extract/batch
Content-Type: application/json

{
  "documents": [
    {"file_url": "...", "document_type": "hospital_bill"},
    {"file_url": "...", "document_type": "aadhaar"}
  ]
}
```

## Running the Server

### Windows (Recommended)
```bash
cd ML
start_ml_server.bat
```

### Linux/Mac
```bash
cd ML
chmod +x start_ml_server.sh
./start_ml_server.sh
```

### Manual (with full dependency installation)
```bash
cd ML
python -m pip install -r requirements.txt --upgrade
python ml_flask_server.py
```

**Note:** First-time setup requires installing torch (2.0.1) and transformers (4.35.2) which are large packages (~500MB). They are listed in requirements.txt and will be installed with the pip command above.

## Environment Variables
- `ML_SERVER_PORT`: Flask server port (default: 5001)
- `ML_DEBUG`: Debug mode (default: true)
- `ML_CLIENT_TIMEOUT`: Client timeout in seconds (default: 30)
- `ML_SERVER_URL`: ML server URL for Django client (default: http://localhost:5001)

## Integration with Django

The Django backend automatically communicates with the Flask ML server:

1. Django receives `/api/admin/claims/<claim_id>/extract/` request
2. Django calls Flask ML server via `ml_client.py`
3. Flask processes documents and returns extracted data
4. Django stores results in database and returns to frontend

## File Structure
```
ML/
├── requirements.txt            # Flask + ML dependencies (torch, transformers)
├── src/
│   ├── Inference.py            # LayoutLMv3 model inference engine
│   ├── pipeline.py             # Document extraction pipeline
│   ├── best_model.bin          # Trained LayoutLMv3 model
│   ├── utils.py
│   ├── validation.py
│   └── loader.py
└── kyc/
    ├── kyc_pipeline.py         # KYC document extraction (Aadhaar, PAN)
    ├── aadhar_extractor.py     # Aadhaar OCR extraction
    └── pan_extractor.py        # PAN OCR extractionxtractor.py

```

## Testing

### Test Flask Server
```bash
curl http://localhost:5001/health
```

### Test Document Extraction
```bash
curl -X POST \
  -F "file=@document.jpg" \
  -F "document_type=hospital_bill" \
  http://localhost:5001/extract
```

