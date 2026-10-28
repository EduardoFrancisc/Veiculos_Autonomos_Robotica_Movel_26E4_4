"""Cliente de câmera em TCP ou UDP.
Instalar: python -m pip install -r requirements.txt, na pasta do pacote.
Consulte documentacao/GUIA_DOIS_TERMINAIS.md para ordem de início e exemplos.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rede_laboratorio import servico_cli
if __name__=='__main__':
    servico_cli('cliente')
