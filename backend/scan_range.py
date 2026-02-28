"""
Scan all real voice files in ml/data/real and ml/data/fake
Print min/max/mean centroid_std and bandwidth_std across each set.
"""
import sys, os, json, glob
sys.path.insert(0, '.')
import librosa
import numpy as np

def measure(path):
    try:
        y, sr = librosa.load(path, sr=16000, duration=3, res_type='kaiser_fast')
        if len(y) < 2000:
            return None
        c = float(np.std(librosa.feature.spectral_centroid(y=y, sr=sr)))
        b = float(np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        z = float(np.std(librosa.feature.zero_crossing_rate(y)))
        return {'centroid_std': round(c,1), 'bandwidth_std': round(b,1), 'zcr_std': round(z,5)}
    except:
        return None

results = {}
for label, pattern in [('real', '../ml/data/real/*'), ('fake', '../ml/data/fake/*')]:
    files = glob.glob(pattern)[:50]   # sample 50 from each
    vals = [measure(f) for f in files]
    vals = [v for v in vals if v]
    if not vals:
        results[label] = 'no files'
        continue
    cs = [v['centroid_std'] for v in vals]
    bs = [v['bandwidth_std'] for v in vals]
    zs = [v['zcr_std'] for v in vals]
    results[label] = {
        'n': len(vals),
        'centroid_std': {'min': round(min(cs),1), 'max': round(max(cs),1), 'mean': round(sum(cs)/len(cs),1)},
        'bandwidth_std': {'min': round(min(bs),1), 'max': round(max(bs),1), 'mean': round(sum(bs)/len(bs),1)},
        'zcr_std':       {'min': round(min(zs),5), 'max': round(max(zs),5), 'mean': round(sum(zs)/len(zs),5)},
    }

with open('range_result.json', 'w') as f:
    json.dump(results, f, indent=2)
print('done')
