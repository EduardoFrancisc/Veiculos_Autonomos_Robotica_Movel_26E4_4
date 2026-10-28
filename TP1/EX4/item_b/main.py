import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

PACOTE = Path(__file__).resolve().parents[3] / "scripts_aula" / "Transformacoes_2D_40_Independentes_Comentados" / "exemplos"
sys.path.insert(0, str(PACOTE))

spec = importlib.util.spec_from_file_location("pipe36", PACOTE / "36_pipeline_dataset_cnn.py")
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

# Redireciona saidas para a pasta do TP.
out = Path(__file__).resolve().parent / "resultados"
out.mkdir(parents=True, exist_ok=True)
pipe.OUT = out

# Corrige leitura de imagens em caminhos com acentos (cv2.imread falha no Windows).
DADOS = pipe.DADOS


def read20_acentos():
    files = sorted((DADOS / "camera_frontal_20").glob("frontal_*.jpg"))
    if len(files) != 20:
        raise ValueError(f"Esperadas 20 imagens, presentes {len(files)}")
    with open(DADOS / "metadados_20.csv", encoding="utf-8") as f:
        meta = {r["arquivo"]: r for r in csv.DictReader(f)}
    out_list = []
    for p in files:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        out_list.append((img, meta[p.name], p.name))
    return out_list


pipe.read20 = read20_acentos

# Corrige cv2.imwrite em caminhos com acentos (Windows).
def save_acentos(exemplo, nome, x):
    folder = pipe.OUT / exemplo
    folder.mkdir(parents=True, exist_ok=True)
    arquivo = folder / nome
    ok, dados = cv2.imencode(arquivo.suffix if arquivo.suffix else ".png", x)
    if not ok:
        raise OSError(arquivo)
    dados.tofile(str(arquivo))
    print("Salvo:", arquivo)
    return arquivo

pipe.save = save_acentos

if __name__ == "__main__":
    pipe.main()
    print("EX4 item B concluido em:", out / pipe.TAG)