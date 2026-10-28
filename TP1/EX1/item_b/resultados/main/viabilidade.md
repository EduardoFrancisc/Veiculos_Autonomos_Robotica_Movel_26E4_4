DISCUSSAO DE VIABILIDADE - Raspberry Pi 4 (4 GB, Cortex-A72)
Medicao neste PC: loop medio = 10.22 ms/frame -> 97.9 FPS.
Etapa dominante: bilateral (1.69 ms), seguida de resize (2.54 ms).
Estimativa conservadora para Pi 4 (fator 5.0x): 51.08 ms/frame -> 19.6 FPS.
Meta de tempo real adotada: 33.33 ms/frame (30 FPS).
Conclusao: NÃO VIÁVEL no Pi 4 para 30 FPS nas condicoes medidas.
Observacoes:
 - Medicao em PC nao equivale a Pi; executar na placa para confirmar (ex. 35_raspberry_pi.py).
 - Se nao viavel: reduzir resolucao de entrada, diminuir d/sigma do bilateral, usar cv2.setNumThreads,
   desabilitar gravacao (gravar=False) e/ou baixar a meta de FPS.
 - Gravacao de video e impressao por frame (verbose) consomem tempo e nao devem contar para FPS de capacidade.
