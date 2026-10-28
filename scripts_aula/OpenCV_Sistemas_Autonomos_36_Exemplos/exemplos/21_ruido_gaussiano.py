"""Exemplo 21 — Adicionar ruído com semente conhecida

COMENTÁRIO GERAL
Cria ruído gaussiano de desvio 12 com um gerador NumPy reproduzível, soma em float, limita a faixa e volta a uint8. A imagem limpa é a referência para os exemplos seguintes.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/21_ruido_gaussiano.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/21_ruido_gaussiano/.

O QUE OBSERVAR
A semente mantém o mesmo ruído para comparar filtros. O ruído é artificial.
"""
import numpy as np
from apoio import (iniciar, carregar_imagem, cinza, salvar_imagem, painel)

a,out=iniciar(__file__)
y=cinza(carregar_imagem(a.imagem));rng=np.random.default_rng(a.seed)
ruido=rng.normal(0,12,y.shape)
suja=np.clip(y.astype(np.float32)+ruido,0,255).astype(np.uint8)
print('Seed:',a.seed,'Desvio do ruído gerado:',float(ruido.std()))
salvar_imagem(out/'ruidosa.png',suja)
painel([y,suja],['Referência','Ruído gaussiano artificial: σ = 12'],out/'ruido.png',a)
