import sys
sys.path.insert(0, "TP2/EX4/item_b")
import main, json

m = main.carregar_yolo()
res = []
for skip, imgsz, nome in [(2, 640, "skip2_img640"),
                          (1, 320, "skip1_img320"),
                          (2, 320, "skip2_img320")]:
    _, r, _ = main.executar_experimento(
        steps=100, render=True, yolo=True, skip=skip, imgsz=imgsz,
        prefix=nome, modelo=m)
    res.append(r)

with open("TP2/EX4/item_b/resultados/main/bench.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("BENCH DONE")
