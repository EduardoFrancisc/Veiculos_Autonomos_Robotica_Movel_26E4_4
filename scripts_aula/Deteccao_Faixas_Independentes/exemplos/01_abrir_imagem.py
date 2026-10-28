"""EXEMPLO 01/32 - Carregar uma imagem de estrada

OBJETIVO: Carregar uma imagem de estrada.
COMO FUNCIONA: cv2.imread abre o PNG do pacote e cv2.imwrite salva uma copia; imprime as dimensoes.
ENTRADA: dados/estrada_sintetica.png (amostra didatica sintetica).
SAIDA ESPERADA: saida/01_abrir_imagem.png e dimensoes no terminal.
COMO EXECUTAR: python exemplos/01_abrir_imagem.py
EXPERIMENTE: Troque o caminho do arquivo de entrada por outra fotografia.
DEPENDENCIAS: cv2, numpy; nao importa nenhum outro arquivo .py.
"""

from pathlib import Path
import cv2
import numpy as np

# Os caminhos sao relativos a este arquivo, nao ao diretorio do Colab.
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_IMAGEM = RAIZ / "dados" / "estrada_sintetica.png"
PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Leitura real de um arquivo PNG com OpenCV.
imagem = cv2.imread(str(CAMINHO_IMAGEM))
if imagem is None:
    raise FileNotFoundError(f"Nao foi possivel ler: {CAMINHO_IMAGEM}")
altura, largura = imagem.shape[:2]

# Este primeiro exemplo de fato abre uma imagem do disco.
print(f"Imagem carregada: {largura} x {altura} pixels")
print(f"Quantidade de canais: {imagem.shape[2]}")
saida = PASTA_SAIDA / "01_abrir_imagem.png"
cv2.imwrite(str(saida), imagem)
print(f"Arquivo salvo: {saida}")
