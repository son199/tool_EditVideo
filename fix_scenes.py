import json
from pathlib import Path

p = Path(r"c:\Users\VT COM\Desktop\YT_POV\video22\scene.json")
data = json.loads(p.read_text(encoding="utf-8"))

fixed = []
new_num = 1
for item in data:
    new_item = {"scene": new_num, "script": item["script"]}
    fixed.append(new_item)
    new_num += 1

p.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Done: {len(fixed)} scenes, numbered 1 to {fixed[-1]['scene']}")
