"""Exemplo 01 — Conferir o ambiente de execução

COMENTÁRIO GERAL
Importa as bibliotecas, identifica CPU, sistema e versões e registra a configuração. Uma matriz pequena confirma que o OpenCV está funcionando.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/01_ambiente.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/01_ambiente/.

O QUE OBSERVAR
Confirme o interpretador escolhido no VS Code. A configuração será necessária para comparar FPS.
"""
import numpy as np
from apoio import (iniciar, salvar_json, painel, ambiente)

a,out=iniciar(__file__)
print('Ambiente:',ambiente())
img=np.zeros((80,120,3),dtype=np.uint8)
img[:,:60]=(30,30,30);img[:,60:]=(220,220,220)
print('Matriz de teste:',img.shape,img.dtype)
salvar_json(out/'ambiente.json',ambiente())
painel([img],['Teste de leitura da matriz'],out/'teste.png',a)
