"""
Flask ML Server for Document Extraction
Microservice for handling ML-based document processing
"""

from flask import Flask, request, jsonify
import os
import sys
import tempfile
import logging
from typing import Dict, List
import uuid
from datetime import datetime
import requests

# Add current directory to Python path
ML_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ML_DIR)
sys.path.append(os.path.join(ML_DIR, 'src'))
sys.path.append(os.path.join(ML_DIR, 'kyc'))

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLFlaskService:
    """Flask service for ML document extraction"""
    
    def __init__(self):
        self.supported_document_types = ['hospital_bill', 'pharmacy_bill', 'aadhaar', 'pan']
    
    def extract_document_data(self, file_path: str, document_type: str) -> Dict:
        """Extract data from document using appropriate ML extractor"""
        try:
            if document_type not in self.supported_document_types:
                raise ValueError(f"Unsupported document type: {document_type}")
            
            logger.info(f"Processing {document_type} document: {file_path}")
            
            if document_type == 'hospital_bill':
                return self._extract_hospital_bill(file_path)
            elif document_type == 'pharmacy_bill':
                return self._extract_pharmacy_bill(file_path)
            elif document_type in ['aadhaar', 'pan']:
                return self._extract_kyc_document(file_path, document_type)
                
        except Exception as e:
            logger.error(f"Error extracting data from {document_type}: {str(e)}")
            raise

    def extract_document_from_url(self, file_url: str, document_type: str) -> Dict:
        """
        Download one document from signed URL and run extraction.
        """
        import hashlib

        timeout = int(os.environ.get('ML_URL_FETCH_TIMEOUT', 60))
        cache_dir = os.path.join(tempfile.gettempdir(), "ml_image_cache")
        os.makedirs(cache_dir, exist_ok=True)

        url_no_query = file_url.split("?", 1)[0]
        file_hash = hashlib.md5(url_no_query.encode()).hexdigest()
        ext = os.path.splitext(url_no_query)[1] or ".jpg"
        cached_file = os.path.join(cache_dir, file_hash + ext)

        if os.path.exists(cached_file):
            logger.info("Using cached image")
            return self.extract_document_data(cached_file, document_type)

        response = requests.get(file_url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(cached_file, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

        return self.extract_document_data(cached_file, document_type)
    
    def _extract_hospital_bill(self, file_path: str) -> Dict:
        """Extract data from hospital bill using actual ML pipeline"""
        # Use your actual ML pipeline for hospital bills
        from src.pipeline import run_pipeline
        results = run_pipeline(file_path)
        
        # Convert your pipeline output to our format
        normalized = {
            'hospital_name': {
                'value': results.get('hospital_name', 'Not found'),
                'confidence': results.get('hospital_confidence', 0.0)
            },
            'patient_name': {
                'value': results.get('patient_name', 'Not found'),
                'confidence': results.get('patient_confidence', 0.0)
            },
            'total_amount': {
                'value': results.get('total_amount', 'Not found'),
                'confidence': results.get('amount_confidence', 0.0)
            },
            'date': {
                'value': results.get('date', 'Not found'),
                'confidence': results.get('date_confidence', 0.0)
            },
            'address': {
                'value': results.get('address', 'Not found'),
                'confidence': results.get('address_confidence', 0.0)
            }
        }
        
        logger.info(f"Hospital bill extracted using ML pipeline: {len(normalized)} fields")
        return normalized
    
    def _extract_pharmacy_bill(self, file_path: str) -> Dict:
        """Extract data from pharmacy bill using the same ML pipeline as hospital bills"""
        # Use the same ML pipeline for pharmacy bills
        from src.pipeline import run_pipeline
        results = run_pipeline(file_path)
        
        # Map hospital pipeline results to pharmacy-specific field names
        normalized = {
            'pharmacy_name': {
                'value': results.get('hospital_name', 'Not found'),  # Use hospital_name for pharmacy
                'confidence': results.get('hospital_confidence', 0.0)
            },
            'patient_name': {
                'value': results.get('patient_name', 'Not found'),
                'confidence': results.get('patient_confidence', 0.0)
            },
            'total_amount': {
                'value': results.get('total_amount', 'Not found'),
                'confidence': results.get('amount_confidence', 0.0)
            },
            'date': {
                'value': results.get('date', 'Not found'),
                'confidence': results.get('date_confidence', 0.0)
            },
            'pharmacy_address': {
                'value': results.get('address', 'Not found'),  # Use address for pharmacy_address
                'confidence': results.get('address_confidence', 0.0)
            }
        }
        
        logger.info(f"Pharmacy bill extracted using ML pipeline: {len(normalized)} fields")
        return normalized
    
    def _extract_kyc_document(self, file_path: str, document_type: str) -> Dict:
        """Extract data from KYC documents using the unified KYC pipeline"""
        # Use the new KYC pipeline that handles both aadhaar and pan
        from kyc.kyc_pipeline import extract_kyc_data
        results = extract_kyc_data(file_path, document_type)
        
        # Handle different document types separately
        if document_type == 'aadhaar':
            # Return only required Aadhaar fields
            normalized = {
                'name': {
                    'value': results.get('name', 'Not found'),
                    'confidence': 0.95 if results.get('name') != 'Not found' else 0.0
                },
                'aadhaar_number': {
                    'value': results.get('aadhaar_number', 'Not found'),
                    'confidence': 0.99 if results.get('aadhaar_number') != 'Not found' else 0.0
                }
            }
            
        elif document_type == 'pan':
            # Return only required PAN fields
            normalized = {
                'name': {
                    'value': results.get('name', 'Not found'),
                    'confidence': 0.97 if results.get('name') != 'Not found' else 0.0
                },
                'pan_number': {
                    'value': results.get('pan_number', 'Not found'),
                    'confidence': 0.98 if results.get('pan_number') != 'Not found' else 0.0
                }
            }
        else:
            raise ValueError(f"Unsupported KYC document type: {document_type}")
        
        logger.info(f"KYC {document_type} extracted using KYC pipeline: {len(normalized)} fields")
        return normalized
    
    def _normalize_extraction_results(self, results) -> Dict:
        """Normalize extraction results to standard format"""
        normalized = {}
        
        if isinstance(results, dict):
            for field_name, data in results.items():
                if isinstance(data, dict) and 'value' in data:
                    normalized[field_name] = {
                        'value': str(data['value']),
                        'confidence': float(data.get('confidence', 0.0))
                    }
                else:
                    normalized[field_name] = {
                        'value': str(data),
                        'confidence': 0.0
                    }
        elif isinstance(results, list):
            for i, item in enumerate(results):
                if isinstance(item, dict):
                    field_name = item.get('field_name', f'field_{i}')
                    normalized[field_name] = {
                        'value': str(item.get('value', '')),
                        'confidence': float(item.get('confidence', 0.0))
                    }
        
        return normalized
    



# Initialize ML service
ml_service = MLFlaskService()


@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint with API information"""
    return jsonify({
        'service': 'ML Document Extraction Server',
        'version': '1.0.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': 'GET /health',
            'extract_single': 'POST /extract',
            'extract_batch': 'POST /extract/batch'
        },
        'supported_document_types': ml_service.supported_document_types,
        'usage': {
            'single_extraction': {
                'method': 'POST',
                'endpoint': '/extract',
                'content_type': 'multipart/form-data',
                'form_data': {
                    'file': 'Document file (image/PDF)',
                    'document_type': 'hospital_bill | pharmacy_bill | aadhaar | pan'
                }
            },
            'batch_extraction': {
                'method': 'POST',
                'endpoint': '/extract/batch',
                'content_type': 'application/json',
                'body': {
                    'documents': [
                        {'file_url': '...', 'document_type': '...'}
                    ]
                }
            }
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Document Extraction',
        'timestamp': datetime.now().isoformat(),
        'supported_types': ml_service.supported_document_types,
        'python_executable': sys.executable
    })


@app.route('/extract', methods=['POST'])
def extract_document():
    """
    Extract data from uploaded document
    
    Expected form data:
    - file: Document file (image/PDF)
    - document_type: Type of document (hospital_bill, pharmacy_bill, aadhaar, pan)
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        if 'document_type' not in request.form:
            return jsonify({'error': 'Document type not specified'}), 400
        
        file = request.files['file']
        document_type = request.form['document_type']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if document_type not in ml_service.supported_document_types:
            return jsonify({
                'error': f'Unsupported document type: {document_type}',
                'supported_types': ml_service.supported_document_types
            }), 400
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(file.filename)[1]
        )
        file.save(temp_file.name)
        temp_file.close()
        
        try:
            # Extract data using ML service
            extracted_data = ml_service.extract_document_data(temp_file.name, document_type)
            
            # Clean up temporary file
            os.unlink(temp_file.name)
            
            return jsonify({
                'success': True,
                'document_type': document_type,
                'extracted_fields': extracted_data,
                'extraction_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
            
    except Exception as e:
        logger.error(f"Error in document extraction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during extraction',
            'details': str(e)
        }), 500


@app.route('/extract/batch', methods=['POST'])
def extract_batch_documents():
    """
    Extract data from multiple documents
    
    Expected JSON data:
    {
        "documents": [
            {"file_url": "...", "document_type": "hospital_bill"},
            {"file_url": "...", "document_type": "aadhaar"}
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'documents' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        documents = data['documents']
        results = {}
        errors = {}
        
        for i, doc in enumerate(documents):
            try:
                file_url = doc.get('file_url')
                document_type = doc.get('document_type')
                
                if not file_url or not document_type:
                    errors[f'document_{i}'] = 'Missing file_url or document_type'
                    continue
                
                if document_type not in ml_service.supported_document_types:
                    errors[f'document_{i}'] = f'Unsupported document type: {document_type}'
                    continue
                
                extracted_data = ml_service.extract_document_from_url(file_url, document_type)
                results[f'document_{i}'] = {
                    'document_type': document_type,
                    'extracted_fields': extracted_data
                }
                
            except Exception as e:
                errors[f'document_{i}'] = str(e)
        
        return jsonify({
            'success': True,
            'results': results,
            'errors': errors,
            'batch_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during batch extraction',
            'details': str(e)
        }), 500


@app.route('/extract/url', methods=['POST'])
def extract_document_from_url():
    """
    Extract data from a document accessible by signed URL.

    Expected JSON:
    {
      "file_url": "https://...signed...",
      "document_type": "hospital_bill|pharmacy_bill|aadhaar|pan"
    }
    """
    try:
        data = request.get_json() or {}
        file_url = data.get('file_url')
        document_type = data.get('document_type')

        if not file_url:
            return jsonify({'error': 'file_url is required'}), 400
        if not document_type:
            return jsonify({'error': 'document_type is required'}), 400
        if document_type not in ml_service.supported_document_types:
            return jsonify({
                'error': f'Unsupported document type: {document_type}',
                'supported_types': ml_service.supported_document_types
            }), 400

        extracted_data = ml_service.extract_document_from_url(file_url, document_type)
        return jsonify({
            'success': True,
            'document_type': document_type,
            'extracted_fields': extracted_data,
            'extraction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in URL extraction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during URL extraction',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('ML_SERVER_PORT', 5001))
    debug = os.environ.get('ML_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting ML Flask server on port {port}")
    logger.info(f"Supported document types: {ml_service.supported_document_types}")
    
    run_kwargs = {
        'host': '0.0.0.0',
        'port': port,
        'debug': debug,
    }
    # Werkzeug on Windows does not support forking-based processes.
    if os.name == 'nt':
        run_kwargs['threaded'] = True
    else:
        run_kwargs['threaded'] = False
        run_kwargs['processes'] = 4

    app.run(**run_kwargs)

