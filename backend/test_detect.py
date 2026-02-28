import sys, json
sys.path.insert(0, '.')
from detect import detect_clone

results = []
for label, f in [('REAL', 'real_voice.wav'), ('CLONE', 'cloned_voice.wav')]:
    r = detect_clone(f)
    verdict_ok = (
        (label == 'REAL'  and not r['is_clone']) or
        (label == 'CLONE' and     r['is_clone'])
    )
    results.append({
        'file': label,
        'verdict': r['verdict'],
        'confidence': r['confidence_score'],
        'pass': verdict_ok,
        'reasons': r['reasons'],
        'method': r['method'],
    })

with open('test_result.json', 'w') as f:
    json.dump(results, f, indent=2)
print('done')
