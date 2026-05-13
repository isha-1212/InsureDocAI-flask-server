
import os
import sys
import logging
import asyncio
from typing import Dict, List
from asgiref.sync import sync_to_async

ML_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from django.db import transaction
from api.models_claim import Claim
from api.models_document import ClaimDocument, ClaimExtractedField
# Import ML client
from ml_client import ml_client

logger = logging.getLogger(__name__)


class MLExtractionService:
    
    
    def __init__(self):
        self.max_concurrency = int(os.environ.get('ML_EXTRACTION_CONCURRENCY', 4))
        self.supported_document_types = {'hospital_bill', 'pharmacy_bill', 'aadhaar', 'pan'}
        self.no_ocr_document_types = {'birth_certificate'}

    @staticmethod
    def _is_connection_error_message(message: str) -> bool:
        text = (message or "").lower()
        markers = (
            "all connection attempts failed",
            "connection refused",
            "connecterror",
            "failed to establish a new connection",
            "unable to reach ml server",
            "name or service not known",
            "temporary failure in name resolution",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _has_real_fields(fields: List[Dict]) -> bool:
        return any(f.get('field_name') != 'extraction_error' for f in fields)

    @staticmethod
    def _is_missing_extracted_value(value) -> bool:
        if value is None:
            return True
        text = str(value).strip().lower()
        if not text:
            return True
        missing_markers = {
            "not found",
            "n/a",
            "na",
            "none",
            "null",
            "unknown",
            "not available",
        }
        return text in missing_markers or text.startswith("not found")

    @staticmethod
    def _filter_fields_for_document_type(document_type: str, fields: List[Dict]) -> List[Dict]:
        allowed = {
            'aadhaar': {'name', 'aadhaar_number'},
            'pan': {'name', 'pan_number'},
        }.get(document_type)
        if not allowed:
            return fields
        return [f for f in fields if f.get('field_name') in allowed or f.get('field_name') == 'extraction_error']
    
    def extract_claim_data(self, claim_id: str) -> Dict:
        """
        Main method to extract data from claim documents
        Returns cached data if available, otherwise performs ML extraction via Flask server
        """
        return asyncio.run(self.extract_claim_data_async(claim_id))

    async def extract_claim_data_async(self, claim_id: str) -> Dict:
        """
        Async version for non-blocking Django/ASGI execution.
        """
        try:
            # Validate claim exists
            await sync_to_async(self._validate_claim, thread_sensitive=True)(claim_id)

            # Get claim documents
            documents = await sync_to_async(self._get_claim_documents, thread_sensitive=True)(claim_id)
            if not documents:
                return self._format_response(claim_id, {}, "no_documents")

            # Check existing cached extraction
            cached_data = await sync_to_async(self._get_cached_extraction, thread_sensitive=True)(claim_id)
            required_types = {d.document_type for d in documents if d.document_type in self.supported_document_types}
            missing_types = {
                doc_type for doc_type in required_types
                if doc_type not in cached_data or not self._is_doc_extraction_complete(cached_data.get(doc_type, []))
            }

            if cached_data and not missing_types:
                logger.info(f"Returning complete cached extraction data for claim {claim_id}")
                return self._format_response(claim_id, cached_data, "cached")

            supported_documents = [d for d in documents if d.document_type in self.supported_document_types]
            documents_to_process = [d for d in supported_documents if d.document_type in missing_types] if missing_types else supported_documents
            
            # Perform ML extraction via Flask server (async + concurrent)
            extracted_data = await self._perform_ml_extraction_via_server_async(claim_id, documents_to_process)
            merged_data = {**cached_data, **extracted_data}

            has_errors = any(
                not self._has_real_fields(fields)
                for fields in extracted_data.values()
            ) if extracted_data else False
            return self._format_response(claim_id, merged_data, "error" if has_errors else "completed")
            
        except Exception as e:
            logger.error(f"Error extracting claim data for {claim_id}: {str(e)}")
            raise
    
    def _validate_claim(self, claim_id: str) -> Claim:
        """Validate that claim exists"""
        try:
            return Claim.objects.get(claim_id=claim_id)
        except Claim.DoesNotExist:
            raise ValueError(f"Claim with ID {claim_id} does not exist")

    def _get_claim_document(self, claim_id: str, document_id: str) -> ClaimDocument:
        """Validate and fetch one document for a claim"""
        try:
            return ClaimDocument.objects.get(claim_id=claim_id, document_id=document_id)
        except ClaimDocument.DoesNotExist:
            raise ValueError(f"Document {document_id} not found for claim {claim_id}")
    
    def _get_cached_extraction(self, claim_id: str) -> Dict:
        """Check if extraction data already exists in database"""
        cached_fields = ClaimExtractedField.objects.filter(claim_id=claim_id)
        
        if not cached_fields.exists():
            return {}

        latest_uploads = {}
        for doc in ClaimDocument.objects.filter(claim_id=claim_id).only('document_type', 'uploaded_at'):
            if not doc.uploaded_at:
                continue
            current = latest_uploads.get(doc.document_type)
            if current is None or doc.uploaded_at > current:
                latest_uploads[doc.document_type] = doc.uploaded_at
        
        # Group cached data by document type
        grouped_data = {}
        for field in cached_fields:
            doc_type = field.document_type
            latest_upload = latest_uploads.get(doc_type)
            if latest_upload and field.created_at and field.created_at < latest_upload:
                continue
            if doc_type not in grouped_data:
                grouped_data[doc_type] = []

            value = field.field_value
            raw_confidence = float(field.confidence_score) if field.confidence_score else 0.0
            confidence = 0.0 if self._is_missing_extracted_value(value) else raw_confidence

            grouped_data[doc_type].append({
                'field_name': field.field_name,
                'value': value,
                'confidence': confidence
            })
        for doc_type in list(grouped_data.keys()):
            grouped_data[doc_type] = self._filter_fields_for_document_type(doc_type, grouped_data[doc_type])
        return grouped_data

    def _is_doc_extraction_complete(self, extracted_fields: List[Dict]) -> bool:
        """
        A document is complete only if it has at least one non-error extracted field.
        """
        if not extracted_fields:
            return False
        for field in extracted_fields:
            if field.get('field_name') != 'extraction_error':
                return True
        return False
    
    def _get_claim_documents(self, claim_id: str) -> List[ClaimDocument]:
        """Fetch all documents for a claim"""
        documents = list(ClaimDocument.objects.filter(claim_id=claim_id))
        logger.info(f"Found {len(documents)} documents for claim {claim_id}")
        for doc in documents:
            logger.info(f"Document: {doc.document_type} - URL: {doc.file_url}")
        return documents
    
    async def _perform_ml_extraction_via_server_async(self, claim_id: str, documents: List[ClaimDocument]) -> Dict:
        """Perform ML extraction using Flask ML server with real Supabase documents (async + concurrent)"""
        extracted_data: Dict = {}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_doc(document: ClaimDocument):
            async with semaphore:
                try:
                    logger.info(f"Extracting {document.document_type} via ML server for claim {claim_id}")

                    supabase_url = self._get_supabase_download_url(
                        document.file_url,
                        document.document_type
                    )

                    logger.info(f"Using Supabase URL: {supabase_url}")

                    extraction_results = await ml_client.extract_from_supabase_url_async(
                        supabase_url,
                        document.document_type
                    )

                    normalized_results = self._normalize_ml_server_results(extraction_results, document.document_type)

                    await asyncio.to_thread(
                        self._store_extraction_results,
                        claim_id,
                        document.document_type,
                        normalized_results
                    )

                    extracted_data[document.document_type] = normalized_results
                    logger.info(f"Successfully extracted {len(normalized_results)} fields for {document.document_type}")
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Error processing document {document.document_id} via ML server: {err_msg}")
                    logger.warning(
                        f"In-process fallback is disabled. Returning extraction_error for document {document.document_id}"
                    )
                    extracted_data[document.document_type] = [{
                        'field_name': 'extraction_error',
                        'value': err_msg,
                        'confidence': 0.0
                    }]

        tasks = [_process_doc(doc) for doc in documents]
        if tasks:
            await asyncio.gather(*tasks)
        return extracted_data
    
    def _get_supabase_download_url(self, file_url: str, document_type: str) -> str:
        """
        Convert stored file_url to proper downloadable Supabase URL
        The file_url is stored as 'bucket_name/path' format, need to create signed URL
        """
        try:
            from api.supabase_client import create_signed_url, get_public_url
            
            # Parse the stored URL format: "bucket_name/user_id/filename"
            # e.g., "hospital_bills/user123/hospital_bill_20240206_abc123.jpg"
            parts = file_url.split('/', 1)  # Split on first slash only
            if len(parts) != 2:
                raise ValueError(f"Invalid file_url format: {file_url}")
            
            bucket_name = parts[0]
            file_path = parts[1]
            
            logger.info(f"Creating signed URL for bucket: {bucket_name}, path: {file_path}")

            # Create signed URL that expires in 1 hour (3600 seconds)
            signed_url_data = create_signed_url(bucket_name, file_path, expires_in=3600)

            # Extract the signed URL from response
            signed_url = signed_url_data.get('signedURL') or signed_url_data.get('signed_url')

            if not signed_url:
                raise ValueError(f"Failed to create signed URL for {file_url}")

            logger.info(f"Created signed URL: {signed_url[:100]}...")
            return signed_url

        except Exception as e:
            logger.error(f"Error creating Supabase download URL for {file_url}: {str(e)}")
            # Fallback to public URL attempt (may not work for private buckets)
            try:
                parts = file_url.split('/', 1)
                if len(parts) == 2:
                    return get_public_url(parts[0], parts[1])
            except Exception:
                pass
            raise
    
    def _normalize_ml_server_results(self, results: Dict, document_type: str) -> List[Dict]:
        """Normalize ML server results to our expected format"""
        normalized = []
        
        for field_name, data in results.items():
            if isinstance(data, dict) and 'value' in data:
                value = str(data['value'])
                confidence = self._normalize_confidence(data.get('confidence', 0.0))
                if self._is_missing_extracted_value(value):
                    confidence = 0.0
                normalized.append({
                    'field_name': field_name,
                    'value': value,
                    'confidence': confidence
                })
            else:
                normalized.append({
                    'field_name': field_name,
                    'value': str(data),
                    'confidence': 0.0
                })
        return self._filter_fields_for_document_type(document_type, normalized)

    def _normalize_confidence(self, value) -> float:
        """
        Normalize confidence score into [0.0, 1.0].
        Accepts model outputs in either 0..1 or 0..100 scale.
        """
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0

        # Handle percentages from some pipelines, e.g. 87.5
        if conf > 1.0 and conf <= 100.0:
            conf = conf / 100.0

        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return conf
    
    def _store_extraction_results(self, claim_id: str, document_type: str, 
                                results: List[Dict]) -> None:
        """Store extraction results in claim_extracted_fields table"""
        field_names = {r.get('field_name') for r in results if r.get('field_name')}

        # Keep DB in sync with latest extraction output for this claim+document_type.
        # Remove stale fields that are no longer present in the latest extraction.
        if field_names:
            ClaimExtractedField.objects.filter(
                claim_id=claim_id,
                document_type=document_type
            ).exclude(field_name__in=field_names).delete()

        has_real_fields = any(r.get('field_name') != 'extraction_error' for r in results)
        if has_real_fields:
            # Clear stale previous error marker when retry eventually succeeds.
            ClaimExtractedField.objects.filter(
                claim_id=claim_id,
                document_type=document_type,
                field_name='extraction_error'
            ).delete()

        for result in results:
            ClaimExtractedField.objects.update_or_create(
                claim_id=claim_id,
                document_type=document_type,
                field_name=result['field_name'],
                defaults={
                    'field_value': result['value'],
                    'confidence_score': result['confidence']
                }
            )
        
        logger.info(f"Stored {len(results)} extracted fields for {document_type} in claim {claim_id}")
    
    def _format_response(self, claim_id: str, extracted_data: Dict, 
                        status: str) -> Dict:
        """Format the final API response"""
        return {
            'claim_id': claim_id,
            'extraction_status': status,
            'documents': extracted_data
        }

    async def extract_claim_document_async(self, claim_id: str, document_id: str) -> Dict:
        """
        Extract fields for one specific claim document.
        """
        await sync_to_async(self._validate_claim, thread_sensitive=True)(claim_id)
        document = await sync_to_async(self._get_claim_document, thread_sensitive=True)(claim_id, document_id)

        if document.document_type in self.no_ocr_document_types:
            return {
                'claim_id': claim_id,
                'document_id': str(document.document_id),
                'document_type': document.document_type,
                'extraction_status': 'skipped_no_ocr',
                'fields': [],
            }

        if document.document_type not in self.supported_document_types:
            return {
                'claim_id': claim_id,
                'document_id': str(document.document_id),
                'document_type': document.document_type,
                'extraction_status': 'unsupported_document_type',
                'fields': [],
            }

        cached_data = await sync_to_async(self._get_cached_extraction, thread_sensitive=True)(claim_id)
        doc_cached_fields = cached_data.get(document.document_type, [])
        if self._is_doc_extraction_complete(doc_cached_fields):
            return {
                'claim_id': claim_id,
                'document_id': str(document.document_id),
                'document_type': document.document_type,
                'extraction_status': 'cached',
                'fields': doc_cached_fields,
            }

        extracted_map = await self._perform_ml_extraction_via_server_async(claim_id, [document])
        doc_fields = extracted_map.get(document.document_type, [])
        has_real_fields = self._has_real_fields(doc_fields)
        return {
            'claim_id': claim_id,
            'document_id': str(document.document_id),
            'document_type': document.document_type,
            'extraction_status': 'completed' if has_real_fields else ('error' if doc_fields else 'no_data'),
            'fields': doc_fields,
        }


# Helper function for Django views to use
async def extract_claim_fields(claim_id: str, async_mode: bool = False) -> Dict:
    """
    Main entry point for ML extraction from Django views
    Uses Flask ML server as microservice
    """
    service = MLExtractionService()
    if async_mode:
        return await service.extract_claim_data_async(claim_id)
    return service.extract_claim_data(claim_id)


async def extract_claim_document(claim_id: str, document_id: str, async_mode: bool = False) -> Dict:
    """
    Extract one document for a claim.
    """
    service = MLExtractionService()
    if async_mode:
        return await service.extract_claim_document_async(claim_id, document_id)
    return asyncio.run(service.extract_claim_document_async(claim_id, document_id))
