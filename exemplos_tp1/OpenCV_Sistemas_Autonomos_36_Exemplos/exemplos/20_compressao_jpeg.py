"""Exemplo 20 — Observar efeitos da compressão JPEG

COMENTÁRIO GERAL
Codifica e decodifica a mesma imagem com qualidades 95, 50 e 10. Mede bytes do JPEG, MSE e PSNR em relação à imagem de entrada, que já pode ter histórico de compressão.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/20_compressao_jpeg.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/20_compressao_jpeg/.

O QUE OBSERVAR
Compare o zoom das bordas. PSNR mede diferença de pixels, não acurácia da percepção.
"""
import cv2
import numpy as np
from apoio import (iniciar, carregar_imagem, salvar_csv, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);imagens=[img];titulos=['Entrada'];linhas=[]
for q in [95,50,10]:
 ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q])
 if not ok: raise OSError('JPEG não codificado.')
 dec=cv2.imdecode(buf,cv2.IMREAD_COLOR)
 mse=float(np.mean((img.astype(np.float32)-dec.astype(np.float32))**2))
 psnr=float('inf') if mse==0 else 10*np.log10(255**2/mse)
 r={'qualidade_jpeg':q,'bytes_jpeg':len(buf),'mse':mse,'psnr_db':float(psnr)}
 linhas.append(r);print(r);imagens.append(dec);titulos.append(f'JPEG Q={q} / {len(buf):,} bytes')
 (out/f'q{q}.jpg').write_bytes(buf.tobytes())
salvar_csv(out/'jpeg.csv',linhas)
painel(imagens,titulos,out/'jpeg.png',a,colunas=2)
h,w=img.shape[:2];zoom=[im[h//3:2*h//3,w//3:2*w//3] for im in imagens]
painel(zoom,titulos,out/'jpeg_zoom.png',a,colunas=2)
