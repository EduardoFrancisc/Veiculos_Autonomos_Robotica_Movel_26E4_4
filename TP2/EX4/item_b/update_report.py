import json

d = json.load(open("TP2/EX4/item_b/resultados/main/bench.json", encoding="utf-8"))
lines = ["", "## Benchmarks de otimizacao (100 steps)", ""]
for r in d:
    lines.append("- %s: %.2f FPS (skip=%d, imgsz=%d, lat=%.1f ms)" %
                 (r["prefix"], r["fps_efetivo"], r["skip"], r["imgsz"], r["latencia_media_ms"]))
p = "TP2/EX4/item_b/resultados/main/relatorio_ex4_item_b.md"
s = open(p, encoding="utf-8").read()
if "## Benchmarks" not in s:
    s += "\n" + "\n".join(lines) + "\n"
open(p, "w", encoding="utf-8").write(s)
print("updated")
