"""Exemplo 10 — Medir o brilho por quadrante

COMENTÁRIO GERAL
Converte para cinza, divide os eixos na metade e calcula a média de cada quadrante. As fatias incluem todos os pixels mesmo quando as dimensões são ímpares.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/10_brilho_quadrantes.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/10_brilho_quadrantes/.

O QUE OBSERVAR
Uma diferença entre quadrantes pode vir da cena, como céu e pista, e não apenas da iluminação.
"""
from apoio import (iniciar, carregar_imagem, cinza, quadrantes, salvar_csv, painel)

a,out=iniciar(__file__)
img=carregar_imagem(a.imagem);y=cinza(img);q=quadrantes(y)
linhas=[{'quadrante':nome,'brilho_medio':float(reg.mean()),'pixels':reg.size} for nome,reg in q.items()]
for r in linhas: print(r)
assert sum(r['pixels'] for r in linhas)==y.size
salvar_csv(out/'quadrantes.csv',linhas)
painel(list(q.values()),[f'{r["quadrante"]}\nMédia {r["brilho_medio"]:.1f}' for r in linhas],out/'quadrantes.png',a,colunas=2)
