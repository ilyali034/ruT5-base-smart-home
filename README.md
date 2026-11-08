# ruT5-base-smart-home 

Safetensors: [huggingface](https://huggingface.co/ilyali034/ruT5-base-smart-home)

```python
from huggingface_hub import snapshot_download
import sys

path = snapshot_download("ilyali034/ruT5-base-smart-home")
sys.path.insert(0, path)
from inference import SmartHomeModel

model = SmartHomeModel(path)
print(model.predict("включи свет на кухне"))
```

## Results (test, 1290 examples)

| Metric | Value |
|---|---|
| Exact match (canonical JSON) | 0.828 |
| Valid JSON rate | 1.000 |
| Function name F1 (micro) | 0.993 |
| Main function accuracy | 0.997 |
| Arguments accuracy | 0.792 |
| Token F1 | 0.956 |
| BLEU | 97.3 |
| chrF | 96.9 |
