import sys
sys.path.insert(0, "backend")
from detect import detect_clone

for label, path in [("REAL", "backend/real_voice.wav"), ("CLONED", "backend/cloned_voice.wav")]:
    r = detect_clone(path)
    print(f"--- {label} ---")
    print(f"  verdict   : {r['verdict']}")
    print(f"  confidence: {r['confidence_score']}")
    print(f"  is_clone  : {r['is_clone']}")
    print(f"  reasons   : {r.get('reasons', [])}")
    print(f"  method    : {r.get('method', 'n/a')}")
    print()
