"""Exemplo 35 — Registrar condições da plataforma embarcada

COMENTÁRIO GERAL
Salva modelo da placa, CPU, versões e leituras opcionais de temperatura, clock e throttling via vcgencmd. Fora do Raspberry Pi, registra que a informação não está disponível.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/35_raspberry_pi.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/35_raspberry_pi/.

O QUE OBSERVAR
Ausência de leitura não significa temperatura baixa ou ausência de throttling. Execute na placa para coletar os dados.
"""
from apoio import (iniciar, salvar_json, ambiente)

import shutil
import subprocess
a,out=iniciar(__file__)
r=ambiente();comando=shutil.which('vcgencmd')
for nome,args in [('temperatura',['measure_temp']),('clock_arm',['measure_clock','arm']),('throttling',['get_throttled'])]:
 if comando:
  try:
   p=subprocess.run([comando,*args],capture_output=True,text=True,timeout=5,check=False)
   r[nome]=p.stdout.strip() if p.returncode==0 else 'Falha na leitura: '+p.stderr.strip()
  except (OSError,subprocess.TimeoutExpired) as e: r[nome]='Indisponível: '+str(e)
 else: r[nome]='vcgencmd não disponível neste sistema'
salvar_json(out/'condicoes_hardware.json',r)
print(r)
print('Roteiro: medir por alguns minutos; não afirmar viabilidade do Pi com FPS de PC/Colab.')
