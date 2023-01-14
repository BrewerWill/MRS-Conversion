def _vax_to_ieee_single_float(data):
  f = []
  nfloat = int(len(data) / 4)

  for i in range(nfloat):

    byte2 = data[0 + i*4]
    byte1 = data[1 + i*4]
    byte4 = data[2 + i*4]
    byte3 = data[3 + i*4]

    # hex 0x80 = binary mask 10000000
    # hex 0x7f = binary mask 01111111

    sign = (byte1 & 0x80) >> 7
    expon = ((byte1 & 0x7f) << 1) + ((byte2 & 0x80) >> 7)
    fract = ((byte2 & 0x7f) << 16) + (byte3 << 8) + byte4

    if sign == 0:
      sign_mult = 1.0
    else:
      sign_mult = -1.0

    if 0 < expon:
      # note 16777216.0 == 2^24
      val = sign_mult * (0.5 + (fract/16777216.0)) * pow(2.0, expon - 128.0)
      f.append(val)
    elif expon == 0 and sign == 0:
      f.append(0)
    else:
      f.append(0)

  return f