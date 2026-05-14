import sys

path = sys.argv[1] if len(sys.argv) > 1 else "datasets/juz1_nl.txt"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

result = content.replace("\n", "\\n")

with open(path, "w", encoding="utf-8") as f:
    f.write(result)

print(f"Done: replaced newlines with \\n in {path}")
'''
with open(path, "r", encoding="utf-8") as f:
    content = f.read().replace("\\n", "\n")
'''
