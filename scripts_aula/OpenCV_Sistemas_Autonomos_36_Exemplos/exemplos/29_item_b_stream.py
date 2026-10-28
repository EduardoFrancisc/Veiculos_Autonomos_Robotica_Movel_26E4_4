"""Exemplo 29 — Executar o pipeline completo em vídeo

COMENTÁRIO GERAL
Lê um vídeo real, executa resize→HSV→V→bilateral, descarta o aquecimento, imprime ms e FPS por frame e grava um feed lado a lado. Os tempos de cada etapa ficam no CSV.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/29_item_b_stream.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/29_item_b_stream/.

O QUE OBSERVAR
Use --mostrar --sem-gui para um feed ao vivo no desktop; no Colab use o vídeo salvo. O FPS final do loop inclui gravação e impressão.
"""
from apoio import (iniciar, rodar_stream)

a,out=iniciar(__file__)
# Pipeline exigido: 640×480 → HSV → V → bilateral.
# rodar_stream trata VideoCapture/VideoWriter, liberação de recursos e medição.
# A GPU do Colab não acelera automaticamente estas chamadas cv2 na CPU.
linhas,resumo=rodar_stream(a,out,gravar=True,verbose=True)
print('Feed:',out/'feed_processado.mp4')
print('FPS processamento:',resumo['fps_processamento'],'FPS loop:',resumo['fps_loop'])
