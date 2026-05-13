"""
KYC Document Pipeline
Processes both Aadhar and PAN cards to extract information
"""

try:
    # Preferred when imported as package module: kyc.kyc_pipeline
    from .aadhar_extractor import extract_aadhaar_info_from_url
    from .pan_extractor import extract_pan_info_from_url
except ImportError:
    # Fallback for direct script execution from this folder
    from aadhar_extractor import extract_aadhaar_info_from_url
    from pan_extractor import extract_pan_info_from_url
import os
import logging

logger = logging.getLogger(__name__)

# Sample image paths/URLs
AADHAR_IMAGE = r"C:\Users\Isha patel\OneDrive\Documents\SE\MYSYdoc\adhaarcard.jpeg" # Change to your actual path
PAN_IMAGE = r"C:\Users\Isha patel\OneDrive\Documents\SE\MYSYdoc\ishaPanCard.jpeg" # Change to your actual path


def extract_kyc_data(file_path: str, document_type: str):
    """
    Extract data from KYC documents (Aadhaar or PAN) - Entry point for ML service
    
    Args:
        file_path (str): Path to the KYC document image/PDF
        document_type (str): Type of document ('aadhaar' or 'pan')
        
    Returns:
        dict: Extracted fields with values and confidence scores
    """
    logger.info(f"Processing KYC document: {file_path}, type: {document_type}")
    
    if document_type == 'aadhaar':
        return extract_aadhaar_info_from_url(file_path)
    elif document_type == 'pan':
        return extract_pan_info_from_url(file_path)
    else:
        raise ValueError(f"Unsupported KYC document type: {document_type}")



def process_kyc_documents(aadhar_image, pan_image):
    """
    Process both Aadhar and PAN card images
    
    Args:
        aadhar_image: Path or URL to Aadhar card image
        pan_image: Path or URL to PAN card image
    
    Returns:
        dict: Combined results from both extractors
    """
    
    # Pass file paths or URLs directly (extractors now handle both)
    aadhar_input = aadhar_image
    pan_input = pan_image
    
    # 1️⃣ Extract Aadhar information
    aadhar_result = extract_aadhaar_info_from_url(aadhar_input, debug=False)
    
    # 2️⃣ Extract PAN information
    pan_result = extract_pan_info_from_url(pan_input, debug=False)
    
    # 3️⃣ Combine results
    kyc_output = {
        "aadhar": {
            "name": aadhar_result.get("name", "Not found"),
            "dob": aadhar_result.get("dob", "Not found"),
            "aadhaar_number": aadhar_result.get("aadhaar_number", "Not found"),
            "status": "success" if "error" not in aadhar_result else "error",
            "error": aadhar_result.get("error")
        },
        "pan": {
            "pan_number": pan_result.get("pan_number", "Not found"),
            "status": "success" if "error" not in pan_result else "error",
            "message": pan_result.get("message"),
            "error": pan_result.get("error")
        },
        "verification": {
            "all_documents_processed": True,
            "aadhar_extracted": "error" not in aadhar_result,
            "pan_extracted": "error" not in pan_result
        }
    }
    
    return kyc_output


def display_kyc_results(results):
    """
    Display extracted KYC information in a formatted manner
    
    Args:
        results: Dictionary containing extraction results
    """
    # Display Aadhar Results
    print("\nAADHAR CARD")
    print("--------------")
    print(f"{'Name':<20}: {results['aadhar']['name']}")
    print(f"{'Date of Birth':<20}: {results['aadhar']['dob']}")
    print(f"{'Aadhaar Number':<20}: {results['aadhar']['aadhaar_number']}")
    print(f"{'Status':<20}: {results['aadhar']['status']}")
    if results['aadhar']['error']:
        print(f"{'Error':<20}: {results['aadhar']['error']}")
    print("--------------")
    
    # Display PAN Results
    print("\nPAN CARD")
    print("--------------")
    print(f"{'PAN Number':<20}: {results['pan']['pan_number']}")
    print(f"{'Status':<20}: {results['pan']['status']}")
    if results['pan'].get('message'):
        print(f"{'Message':<20}: {results['pan']['message']}")
    if results['pan']['error']:
        print(f"{'Error':<20}: {results['pan']['error']}")


def process_single_document(image_path, doc_type="aadhar"):
    """
    Process a single KYC document (either Aadhar or PAN)
    
    Args:
        image_path: Path or URL to the document image
        doc_type: Type of document ('aadhar' or 'pan')
    
    Returns:
        dict: Extraction results
    """
    
    print(f"\n🔍 Processing {doc_type.upper()} card...")
    
    # Pass file path or URL directly (extractors now handle both)
    input_path = image_path
    
    if doc_type.lower() == "aadhar":
        result = extract_aadhaar_info_from_url(input_path)
        
        print("\n📄 AADHAR EXTRACTION RESULTS")
        print("-"*40)
        print(f"{'Name':<20}: {result.get('name', 'Not found')}")
        print(f"{'Date of Birth':<20}: {result.get('dob', 'Not found')}")
        print(f"{'Aadhaar Number':<20}: {result.get('aadhaar_number', 'Not found')}")
        
        if "error" in result:
            print(f"{'Error':<20}: {result['error']}")
        
        return result
        
    elif doc_type.lower() == "pan":
        result = extract_pan_info_from_url(input_path)
        
        print("\n💳 PAN EXTRACTION RESULTS")
        print("-"*40)
        print(f"{'PAN Number':<20}: {result.get('pan_number', 'Not found')}")
        
        if "message" in result:
            print(f"{'Message':<20}: {result['message']}")
        if "error" in result:
            print(f"{'Error':<20}: {result['error']}")
        
        return result
    
    else:
        return {"error": f"Unknown document type: {doc_type}"}


# ================= RUN =================
if __name__ == "__main__":
    
    # Process both documents together
    kyc_results = process_kyc_documents(AADHAR_IMAGE, PAN_IMAGE)
    display_kyc_results(kyc_results)
    
    
    # Option 2: Process individual documents
    # Uncomment below to test individual processing
    
    # print("\n\n" + "="*60)
    # print("OPTION 2: Processing Individual Documents")
    # print("="*60)
    # 
    # aadhar_only = process_single_document(AADHAR_IMAGE, doc_type="aadhar")
    # pan_only = process_single_document(PAN_IMAGE, doc_type="pan")
