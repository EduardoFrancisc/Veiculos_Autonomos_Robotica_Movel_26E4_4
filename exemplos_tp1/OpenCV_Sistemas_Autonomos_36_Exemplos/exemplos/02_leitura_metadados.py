"""Exemplo 02 — Carregar a imagem e imprimir os metadados

COMENTÁRIO GERAL
Decodifica a imagem frontal como BGR e imprime largura, altura, dtype, canais, tamanho da matriz e tamanho do arquivo. Os dois tamanhos respondem a perguntas diferentes.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/02_leitura_metadados.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/02_leitura_metadados/.

O QUE OBSERVAR
Compare bytes_pixels e bytes_arquivo. A compressão do arquivo não reduz automaticamente a matriz decodificada.
"""
from apoio import (iniciar, carregar_imagem, metadados, salvar_json, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem)
m=metadados(img,a.imagem)
salvar_json(out/'metadados.json',m)
painel([img],['Imagem frontal'],out/'imagem.png',a)
