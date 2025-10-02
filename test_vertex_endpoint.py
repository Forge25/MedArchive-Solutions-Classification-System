from google.cloud import aiplatform

PROJECT_ID = "forge-ml-project" 
ENDPOINT_ID = "2909108755490668544"
LOCATION = "us-central1"

# Initialize Vertex AI
print("Initializing Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Get the endpoint
print(f"Connecting to endpoint: {ENDPOINT_ID}")
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

# Test medical transcriptions
test_cases = [
    {
        "text": "Patient presents with chest pain and shortness of breath. ECG shows ST elevation. Troponin levels elevated. Diagnosis: acute myocardial infarction. Started on aspirin, beta blocker, and heparin drip.",
        "expected": "Cardiovascular / Pulmonary"
    },
    {
        "text": "Patient undergoes arthroscopic knee surgery for torn meniscus. Post-operative recovery appears normal. Physical therapy recommended. Follow-up appointment scheduled in 2 weeks.",
        "expected": "Orthopedic"
    },
    {
        "text": "Neurological examination reveals decreased memory function and difficulty with cognitive tasks. MRI of brain ordered. Possible early dementia. Cognitive assessment recommended.",
        "expected": "Neurology"
    }
]

print("\n" + "=" * 80)
print("VERTEX AI ENDPOINT TEST - MEDARCHIVE SOLUTIONS")
print("=" * 80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test Case {i}:")
    print(f"{'='*80}")
    print(f"\nInput Text:")
    print(f"  {test_case['text']}")
    print(f"\nExpected Specialty: {test_case['expected']}")

    try:
        # Make prediction
        prediction = endpoint.predict(instances=[test_case['text']])

        print(f"\n✅ PREDICTION SUCCESSFUL!")
        print(f"Predicted Specialty: {prediction.predictions[0]}")

        # Check if matches expected
        if prediction.predictions[0] == test_case['expected']:
            print(f"✓ Matches expected result!")
        else:
            print(f"⚠ Different from expected (this is okay, model may vary)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
