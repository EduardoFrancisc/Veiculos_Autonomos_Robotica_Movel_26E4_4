"""Exemplo 28 — Consolidar todas as saídas do item A

COMENTÁRIO GERAL
Executa leitura, metadados, brilho por quadrantes, CLAHE em cinza, histogramas sobrepostos e comparação visual. Também oferece CLAHE em Lab como extensão colorida.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/28_item_a_completo.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/28_item_a_completo/.

O QUE OBSERVAR
Confira os JSON, PNG e o terminal. Use uma imagem frontal KITTI, nuScenes ou própria para a entrega.
"""
from apoio import (iniciar, item_a)

a,out=iniciar(__file__)
# A função item_a reúne as operações estudadas nos exemplos 02, 10, 15 e 16.
# Em sombras, faixas e obstáculos perdem contraste; sob saturação, detalhes
# desaparecem. CLAHE pode ajudar localmente e também realçar ruído. A
# validação final depende da qualidade e do desempenho da percepção.
rel=item_a(a,out)
print('Item A concluído em:',out)
print('Definição de brilho:',rel['definicao_brilho'])
