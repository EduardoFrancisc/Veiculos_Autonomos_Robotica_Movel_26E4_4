"""EXEMPLO 28 — Inspecao de conjunto com 20 imagens

OBJETIVO E CONTEXTO:
O código carrega 20 imagens de câmera frontal com variações controladas de brilho, rotação e escala e monta um painel 4 × 5. Ele também consulta os metadados associados às imagens e confirma a quantidade e a resolução dos quadros. A importância desse exemplo está em verificar a diversidade e a integridade do conjunto de dados antes de iniciar normalização ou treinamento, identificando visualmente quais transformações já estão presentes nas amostras.

COMO A IMPLEMENTACAO FUNCIONA:
Este codigo possui as proprias funcoes de leitura, transformacao e salvamento.
O fluxo parte das imagens fornecidas em dados/, prepara os parametros
matematicos necessarios, executa a operacao estudada, apresenta metricas
ou matrizes quando aplicavel e salva evidencias em saidas/28_vinte_imagens_frontais/.
Cada etapa da funcao main() recebe comentarios proximos das instrucoes
correspondentes para permitir acompanhar a execucao linha a linha.

FUNDAMENTO E INTERPRETACAO:
percorrer os arquivos JPG e apresentar um grid de 4 linhas por 5 colunas.
Analise o resultado visual junto com a matriz ou os indicadores impressos:
uma transformacao correta exige coerencia das coordenadas, resolucao,
espaco de cores e rotulos associados aos dados.

IMPORTANCIA E CUIDADOS:
Inspecao de dataset é necessaria antes de aplicar normalizacao e aumento artificial de dados.

EXERCICIO PARA OS ALUNOS:
Localizar visualmente variacoes de brilho e orientacao.

EXECUCAO:
python exemplos/28_vinte_imagens_frontais.py
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

def read20():
    """Le exatamente 20 quadros frontais e seus metadados CSV.
    Valida a quantidade esperada e devolve tuplas (imagem BGR, metadados, nome).
    O campo steering_simulado_rad e didatico e NAO representa direcao real."""
    import csv
    files = sorted((DADOS / 'camera_frontal_20').glob('frontal_*.jpg'))
    if len(files) != 20:
        raise ValueError(f'Esperadas 20 imagens, presentes {len(files)}')
    with open(DADOS / 'metadados_20.csv', encoding='utf-8') as f:
        meta = {r['arquivo']: r for r in csv.DictReader(f)}
    return [(cv2.imread(str(p)), meta[p.name], p.name) for p in files]

def grid_4x5(images, labels=None, w=252, h=176):
    """Organiza exatamente 20 imagens em quatro linhas e cinco colunas.
    Acrescenta identificadores e devolve uma imagem BGR de visualizacao,
    separada dos tensores float32 usados para treinamento."""
    if len(images) != 20:
        raise ValueError('Grid exige exatamente 20 imagens')
    canvas = np.full((4 * (h + 34), 5 * w, 3), 250, np.uint8)
    for i, a in enumerate(images):
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        x = i % 5 * w
        y = i // 5 * (h + 34)
        canvas[y:y + h, x:x + w] = a
        cv2.putText(canvas, (labels[i] if labels else f'Imagem {i + 1:02d}')[:31], (x + 5, y + h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 45, 60), 1, cv2.LINE_AA)
    return canvas

TAG = '28_vinte_imagens_frontais'

# Fluxo principal explicado por etapas: execute e acompanhe a pasta saidas/.
def main():
    # Le o conjunto de 20 imagens e os metadados; verifica-se a disponibilidade dos arquivos.
    data = read20()

    # Separa apenas as imagens BGR das tuplas que tambem contem os metadados.
    im = [x for x, _, _ in data]

    # Organiza as 20 fotografias em um grid 4x5 com legendas, para inspeção em sala.
    grid = grid_4x5(im, [f'Frame {i + 1:02d}' for i in range(20)])

    # Salva a evidencia visual do processamento na pasta exclusiva deste programa.
    save(TAG, 'grid_4x5.png', grid)

    # Apresenta no terminal os valores para conferencia e comparacao dos resultados.
    print('Lidas', len(im), 'imagens, resolucao', im[0].shape, 'origens reais: video Udacity')

if __name__ == '__main__':
    main()
