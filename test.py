import os

sdat_filename = "../VGH0054_777540_Initial_MRI_WIP_MOTOR_PRESS_135_7_1_raw_act.SDAT"
spar_filename = "../VGH0054_777540_Initial_MRI_WIP_MOTOR_PRESS_135_7_1_raw_act.SPAR"

# Read .SDAT file and extract raw data
with open(sdat_filename, 'rb') as fin:
  raw_bytes = fin.read()
  print(raw_bytes)