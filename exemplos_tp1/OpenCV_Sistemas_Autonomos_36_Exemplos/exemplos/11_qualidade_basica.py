"""Exemplo 11 — Inspecionar regiões escuras e saturadas

COMENTÁRIO GERAL
Calcula as frações com cinza abaixo de 30 e acima de 245 e desenha máscaras. São limiares didáticos, ajustáveis ao contexto, não uma nota universal de qualidade.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/11_qualidade_basica.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/11_qualidade_basica/.

O QUE OBSERVAR
Média alta não significa boa imagem: pixels saturados podem ter perdido informação.
"""
import numpy as np
from apoio import (iniciar, carregar_imagem, cinza, salvar_json, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);y=cinza(img)
escuro=(y<30).astype(np.uint8)*255;claro=(y>245).astype(np.uint8)*255
r={'media':float(y.mean()),'desvio':float(y.std()),'fracao_cinza_menor_30':float((y<30).mean()),'fracao_cinza_maior_245':float((y>245).mean())}
print(r);salvar_json(out/'qualidade.json',r)
painel([y,escuro,claro],['Cinza','Máscara: intensidade < 30','Máscara: intensidade > 245'],out/'qualidade.png',a)
