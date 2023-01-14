import suspect as ss
import numpy as np
import struct
import os
from vax_to_ieee_single_float import _vax_to_ieee_single_float
from philips_orientation import _philips_orientation
from spec2nii.nifti_orientation import NIFTIOrient, calc_affine

spar_types = {
  "floats": ["ap_size", "lr_size", "cc_size", "ap_off_center", "lr_off_center",
               "cc_off_center", "ap_angulation", "lr_angulation", "cc_angulation",
               "image_plane_slice_thickness", "slice_distance", "spec_col_lower_val",
               "spec_col_upper_val", "spec_row_lower_val", "spec_row_upper_val",
               "spectrum_echo_time", "echo_time", "dim1_step"],
    "integers": ["samples", "rows", "synthesizer_frequency", "offset_frequency",
                 "sample_frequency", "echo_nr", "mix_number", "t0_mul_direction",
                 "repetition_time", "averages", "volumes", "dim1_pnts"
                 "volume_selection_method", "nr_of_slices_for_multislice",
                 "spec_num_col", "spec_num_row", "num_dimensions", "TSI_factor",
                 "spectrum_inversion_time", "image_chemical_shift",
                 "t0_mu1_direction"],
    "strings": ["scan_id", "scan_date", "patient_name", "patient_birth_date",
                "patient_position", "patient_orientation", "nucleus",
                "volume_selection_enable", "phase_encoding_enable", "t1_measurement_enable",
                "t2_measurement_enable", "time_series_enable", "Spec.image in plane transf",
                "spec_data_type", "spec_sample_extension", "spec_col_extension",
                "spec_row_extension", "echo_acquisition", "resp_motion_comp_technique",
                "de_coupling", "equipment_sw_verions", "examination_name"],
}

sdat_filename = "./VGH0054_777540_Initial_MRI_WIP_MOTOR_PRESS_135_7_1_raw_act.SDAT"
spar_filename = None

path, ext = os.path.splitext(sdat_filename)
output_filename = path + ".rda"

if spar_filename is None:
  # match the capitalisation of the sdat extension
  if ext == ".SDAT":
    spar_filename = path + ".SPAR"
  elif ext == ".sdat":
    spar_filename = path + ".spar"

# Read SPAR parameters
with open(spar_filename, 'r') as fin:
  spar_params = {}
  for line in fin:
    # ignore empty lines and comments starting with !
    if line != "\n" and not line.startswith("!"):
      key, value = map(str.strip, line.split(":", 1))
      if key in spar_types["floats"]:
        spar_params[key] = float(value)
      elif key in spar_types["integers"]:
        spar_params[key] = int(value)
      elif key in spar_types["strings"]:
        spar_params[key] = value
      else:
        pass

# Dictionary for the RDA header
header_dict = {}

# These parameters are assumed to have these values
header_dict["CSIMatrixSize[0]"] = 1
header_dict["CSIMatrixSize[1]"] = 1
header_dict["CSIMatrixSize[2]"] = 1
header_dict["CSIMatrixSize"] = [1,1,1]
header_dict["VectorSize"] = spar_params["samples"]

data_shape = header_dict["CSIMatrixSize"][::-1]
data_shape.append(header_dict["VectorSize"])
data_shape = np.array(data_shape)

# Define the equivalencies found:
header_dict["NumberOfAverages"] = spar_params["averages"]
header_dict["PatientName"] = spar_params["patient_name"]
header_dict["TR"] = spar_params["repetition_time"]
header_dict["TE"] = spar_params["echo_time"]
header_dict["MRFrequency"] = spar_params["synthesizer_frequency"] * 1e-6
header_dict["DwellTime"] = spar_params["dim1_step"] * 1e6
header_dict["TI"] = spar_params["spectrum_inversion_time"]
# Review this since the encoding is different
header_dict["PatientPosition"] = (spar_params["patient_position"] + spar_params["patient_orientation"]).replace('""','_')
header_dict["SoftwareVersion[0]"] = spar_params["equipment_sw_verions"]
header_dict["InstitutionName"] = "Robarts Research Institute"
# Review this since the encoding is different
header_dict["ProtocolName"] = spar_params["scan_id"]
header_dict["PatientBirthDate"] = spar_params["patient_birth_date"]
header_dict["Nucleus"] = spar_params["nucleus"]

# Get the "spacing" variable in transformation_matrix
header_dict["PixelSpacingCol"] = spar_params["lr_size"]
header_dict["PixelSpacingRow"] = spar_params["ap_size"]
header_dict["PixelSpacing3D"] = spar_params["cc_size"]
voxel_size = (header_dict["PixelSpacingCol"], header_dict["PixelSpacingRow"], header_dict["PixelSpacing3D"])
spacing = list(voxel_size)
spacing.append(1.0)

# Read .SDAT file and extract raw data
with open(sdat_filename, 'rb') as fin:
  raw_bytes = fin.read()

floats = _vax_to_ieee_single_float(raw_bytes)
data_iter = iter(floats)
complex_iter = (complex(r, -i) for r, i in zip(data_iter, data_iter))
raw_data = np.fromiter(complex_iter, "complex64")
raw_data = np.reshape(raw_data, (spar_params["rows"], spar_params["samples"])).squeeze()

# calculate transformation matrix
voxel_size = np.array([spar_params["lr_size"],
  spar_params["ap_size"],
  spar_params["cc_size"]])
position_vector = np.array([spar_params["lr_off_center"],
  spar_params["ap_off_center"],
   spar_params["cc_off_center"]])

A = np.eye(3)
for a,ang in enumerate(["lr_angulation", "ap_angulation", "cc_angulation"]):
  axis = np.zeros(3)
  axis[a] = 1
  A = A @ ss.rotation_matrix(spar_params[ang]/180*np.pi,axis)
e1 = A[:,0]
e1 = e1 / np.linalg.norm(e1)
e2 = A[:,1]
e2 = e2 / np.linalg.norm(e2)

transform = ss.transformation_matrix(e1, e2, position_vector, voxel_size)

# Convert raw data from Philips MRS object
reverse_float = []
for comp in raw_data:
    reverse_float.append(comp.real)
    reverse_float.append(comp.imag)

# Trying to mulitply each float by 10e3 to have a range closer to rda format
# THIS MIGHT BE WRONG
for i in range(len(reverse_float)):
    reverse_float[i]*=1000

data = struct.pack("<{}d".format(np.prod(data_shape) * 2), *reverse_float)

# Extract RowVector, ColumnVector, VOIPositionSag, VOIPositionCor, VOIPositionTra
matrix = transform.copy()
for i in range(4):
  for j in range(4):
    matrix[i, j] /= spacing[j]

x_vector = matrix[:3, 0]
y_vector = matrix[:3, 1]
translation = matrix[:3, 3]

header_dict["VOIPositionSag"] = translation[0]
header_dict["VOIPositionCor"] = translation[1]
header_dict["VOIPositionTra"] = translation[2]

header_dict["RowVector[0]"] = x_vector[0]
header_dict["RowVector[1]"] = x_vector[1]
header_dict["RowVector[2]"] = x_vector[2]

header_dict["ColumnVector[0]"] = y_vector[0]
header_dict["ColumnVector[1]"] = y_vector[1]
header_dict["ColumnVector[2]"] = y_vector[2]

# Calculate position vectors
if spar_params["volume_selection_enable"] == '"yes"':
    affine = _philips_orientation(spar_params)
else:
    # Use default affine
    affine = np.diag(np.array([10000, 10000, 10000, 1]))
orientation = NIFTIOrient(affine)
Q44 = orientation.Q44

header_dict["PositionVector[0]"] = -Q44[0,3]
header_dict["PositionVector[1]"] = -Q44[1,3]
header_dict["PositionVector[2]"] = -Q44[2,3]

out = open(output_filename, "wb")

# Write header
para_list = [
    "PatientName", 
    "PatientID", "PatientSex", 
    "PatientBirthDate", "StudyDate", 
    "StudyTime", "StudyDescription", 
    "PatientAge", "PatientWeight", 
    "SeriesDate", "SeriesTime", 
    "SeriesDescription", "ProtocolName", 
    "PatientPosition", "SeriesNumber", 
    "InstitutionName", "StationName", 
    "ModelName", "DeviceSerialNumber", 
    "SoftwareVersion[0]", "InstanceDate", 
    "InstanceTime", "InstanceNumber", 
    "InstanceComments", "AcquisitionNumber", 
    "SequenceName", "SequenceDescription", 
    "TR", "TE", "TM", "TI", "DwellTime", 
    "EchoNumber", "NumberOfAverages", 
    "MRFrequency", "Nucleus", "MagneticFieldStrength", 
    "NumOfPhaseEncodingSteps", "FlipAngle", "VectorSize", 
    "CSIMatrixSize[0]", "CSIMatrixSize[1]", 
    "CSIMatrixSize[2]", "CSIMatrixSizeOfScan[0]", 
    "CSIMatrixSizeOfScan[1]", "CSIMatrixSizeOfScan[2]", 
    "CSIGridShift[0]", "CSIGridShift[1]", 
    "CSIGridShift[2]", "HammingFilter", 
    "FrequencyCorrection", "TransmitCoil", 
    "TransmitRefAmplitude[1H]", "SliceThickness", 
    "PositionVector[0]", "PositionVector[1]", 
    "PositionVector[2]", "RowVector[0]", "RowVector[1]", 
    "RowVector[2]", "ColumnVector[0]", "ColumnVector[1]", 
    "ColumnVector[2]", "VOIPositionSag", "VOIPositionCor", 
    "VOIPositionTra", "VOIThickness", "VOIPhaseFOV", 
    "VOIReadoutFOV", "VOINormalSag", "VOINormalCor", 
    "VOINormalTra", "VOIRotationInPlane", "FoVHeight", 
    "FoVWidth", "FoV3D", "PercentOfRectFoV", 
    "NumberOfRows", "NumberOfColumns", "NumberOf3DParts", 
    "PixelSpacingRow", "PixelSpacingCol", "PixelSpacing3D"
    ]

out.write(b">>> Begin of header <<<\r\n")
for para in para_list:
    out.write(bytes(para + ":", 'windows-1252'))
    if para in header_dict:
        content = " " + str(header_dict[para])
        out.write(bytes(content, 'windows-1252'))
    else:
        out.write(bytes(" ", 'windows-1252'))
    out.write(b"\r\n")

out.write(b">>> End of header <<<\r\n")

# Write raw data
out.write(data)

# Close output file
out.close()