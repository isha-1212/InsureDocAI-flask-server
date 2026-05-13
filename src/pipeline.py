import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Inference import run_inference
from validation import (
    validate_hospital_name,
    validate_patient_name,
    validate_date,
    validate_address,
    validate_total_amount
)

IMAGE_PATH = r"C:\vscode\CVprojects\SGP6\data\hospital\images\img02.png"

def run_pipeline(image_path):

    # 1️⃣ Run inference (silent)
    inference_result = run_inference(image_path)

    # 2️⃣ Send to all validators
    hospital_result = validate_hospital_name(
        ml_predictions=inference_result["ml_predictions"]
    )
    
    patient_result = validate_patient_name(
        ml_predictions=inference_result["ml_predictions"]
    )
    
    date_result = validate_date(
        ml_predictions=inference_result["ml_predictions"]
    )
    
    address_result = validate_address(
        ml_predictions=inference_result["ml_predictions"]
    )
    
    amount_result = validate_total_amount(
        ml_predictions=inference_result["ml_predictions"]
    )

    # 3️⃣ Final output
    final_output = {
        "hospital_name": hospital_result["hospital_name"],
        "hospital_confidence": hospital_result["confidence"],
        "hospital_status": hospital_result["status"],
        
        "patient_name": patient_result["patient_name"],
        "patient_confidence": patient_result["confidence"],
        "patient_status": patient_result["status"],
        
        "date": date_result["date"],
        "date_confidence": date_result["confidence"],
        "date_status": date_result["status"],
        
        "address": address_result["address"],
        "address_confidence": address_result["confidence"],
        "address_status": address_result["status"],
        
        "total_amount": amount_result["total_amount"],
        "amount_confidence": amount_result["confidence"],
        "amount_status": amount_result["status"]
    }

    return final_output


# ================= RUN =================
if __name__ == "__main__":
    result = run_pipeline(IMAGE_PATH)

    print("\n===== FINAL OUTPUT AFTER VALIDATION =====\n")
    
    print(f"{'HOSPITAL NAME':<20}: {result['hospital_name']}")
    print(f"{'  Confidence':<20}: {result['hospital_confidence']}")
    print(f"{'  Status':<20}: {result['hospital_status']}")
    print()
    
    print(f"{'PATIENT NAME':<20}: {result['patient_name']}")
    print(f"{'  Confidence':<20}: {result['patient_confidence']}")
    print(f"{'  Status':<20}: {result['patient_status']}")
    print()
    
    print(f"{'DATE':<20}: {result['date']}")
    print(f"{'  Confidence':<20}: {result['date_confidence']}")
    print(f"{'  Status':<20}: {result['date_status']}")
    print()
    
    print(f"{'ADDRESS':<20}: {result['address']}")
    print(f"{'  Confidence':<20}: {result['address_confidence']}")
    print(f"{'  Status':<20}: {result['address_status']}")
    print()
    
    print(f"{'TOTAL AMOUNT':<20}: {result['total_amount']}")
    print(f"{'  Confidence':<20}: {result['amount_confidence']}")
    print(f"{'  Status':<20}: {result['amount_status']}")
