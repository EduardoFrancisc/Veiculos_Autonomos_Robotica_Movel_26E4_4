"""EXEMPLO 01 — Metadados da imagem e verificacao de entrada

OBJETIVO E CONTEXTO:
Este programa inspeciona uma imagem da câmera frontal, identificando altura, largura, número de canais, tipo de dados e valores mínimo e máximo dos pixels. Ele utiliza o OpenCV para carregar a imagem e salvar uma cópia, permitindo verificar se os dados estão adequados antes de aplicar qualquer transformação. Essa inspeção é importante porque uma CNN precisa receber imagens em um formato conhecido, e discrepâncias de resolução ou canais podem comprometer o treinamento.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/01_inspecao_da_imagem/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
altura, largura e canais: shape; tipo numerico: dtype; limites: min/max.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
A imagem lida pelo OpenCV costuma estar em BGR uint8, e uma CNN exige uma forma de entrada conhecida.

EXERCICIO PARA OS ALUNOS:
Alterar a imagem e comparar resolucao, tipo numerico e numero de canais.

EXECUCAO:
python exemplos/01_inspecao_da_imagem.py
Nao e necessario executar os demais exemplos antes deste.
As dependencias externas sao cv2 e numpy; o exemplo de CNN
admite TensorFlow opcionalmente. As imagens ficam em dados/."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Resolucao robusta de caminhos: funciona ao executar a partir de qualquer pasta.
BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / 'dados'
OUT = BASE / 'saidas'
OUT.mkdir(parents=True, exist_ok=True)

# Funcoes privativas deste arquivo. Nenhum modulo Python adicional do pacote e necessario.

def img(nome='frontal_kitti.png'):
    """Carrega uma imagem BGR da pasta dados. A leitura usa a raiz baseada em __file__,
    portanto o programa funciona independentemente do diretorio de trabalho.
    Lanca um erro compreensivel se a imagem nao estiver disponivel."""
    p = DADOS / nome
    x = cv2.imread(str(p))
    if x is None:
        raise FileNotFoundError(p)
    return x

def save(exemplo, nome, x):
    """Cria a pasta deste exercicio e salva uma imagem com cv2.imwrite.
    Retorna o caminho e interrompe com erro se a gravacao falhar."""
    folder = OUT / exemplo
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / nome
    if not cv2.imwrite(str(file), x):
        raise OSError(file)
    print('Salvo:', file)
    return file

TAG = '01_inspecao_da_imagem'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Carrega a fotografia BGR original, usada como referencia para comparar antes e depois.
    a = img()

    # Obtem as dimensoes da imagem: o OpenCV organiza primeiro altura (linhas), depois largura
    # (colunas).
    h, w, c = a.shape

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Altura, largura, canais:', h, w, c, 'dtype:', a.dtype, 'intervalo:', int(a.min()), int(a.max()))

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'original.png', a)

if __name__ == '__main__':
    main()
