"""

    An experiment to test 1nn + euclidean distance + various data set
    parameter setting refer paper: Classification of time series by shapelet transformation, Hills, Jon

"""

from joblib import cpu_count
from shapelet.transform_basic import ShapeletTransformSimplicity
from tools import output

from base import OUT_ROOT
from run_shapelet_framework import *

dataset = [('Adiac', 3, 10), ('Beef', 8, 30), ('BeetleFly', 30, 101), ('BirdChicken', 30, 101),
           ('ChlorineConcentration', 7, 20), ('Coffee', 18, 30), ('DiatomSizeReduction', 7, 16),
           ('ECGFiveDays', 24, 76), ('FaceFour', 20, 120), ('Gun_Point', 24, 55),
           ('ItalyPowerDemand', 7, 14), ('Lighting7', 20, 80), ('MedicalImages', 9, 35),
           ('MoteStrain', 16, 31), ('SonyAIBORobotSurface', 15, 36), ('Symbols', 52, 155),
           ('synthetic_control', 20, 56), ('Trace', 62, 232), ('TwoLeadECG', 7, 13),
           ('MiddlePhalanxOutlineAgeGroup', 15, 55), ('MiddlePhalanxOutlineCorrect', 15, 55),
           ('MiddlePhalanxTW', 15, 55),
           ('ProximalPhalanxOutlineAgeGroup', 13, 45), ('ProximalPhalanxOutlineCorrect', 13, 45),
           ('ProximalPhalanxTW', 13, 45),
           ('DistalPhalanxOutlineAgeGroup', 9, 50), ('DistalPhalanxOutlineCorrect', 9, 50),
           ('DistalPhalanxTW', 9, 50)]

# dataset = [('BeetleFly', 30, 101)]

num_cpu = cpu_count()

n_jobs = num_cpu-1
num_shapelet = 0.5
length_increment = 5
position_increment = 10
n_neighbors = 1


str_parameter = "%skk \n euclidean distance \n shapelet ratio: %s \n length increment: %s \n " \
                "position increment: %s \n parallel cpu: %s"\
                % (n_neighbors, num_shapelet, length_increment, position_increment, n_jobs)

print("="*80)
print(__doc__)
print(str_parameter)

str_param_reduced = "shapelet_%skk_ed_%s_%s_%s" % (n_neighbors, num_shapelet, length_increment, position_increment)
filedir_result = os.path.join(OUT_ROOT, str_param_reduced+'.md')
dir_shapelet = os.path.join(OUT_ROOT, str_param_reduced)
if not os.path.exists(dir_shapelet):
    os.makedirs(dir_shapelet)

description = __doc__ + '\n' + str_parameter + '\n'
head = ('name', 'accuracy', 'number of shapelets', 'min length', 'max length')

file_result = open(filedir_result, 'w')
file_result.write(output.headmarkdown(head))

result = [head]
for item in dataset:
    name, min_shapelet_length, max_shapelet_length = item
    acc, best_shapelets = test_shapelet(name,
                                        ShapeletTransformSimplicity,
                                        min_shapelet_length=min_shapelet_length,
                                        max_shapelet_length=max_shapelet_length,
                                        num_shapelet=num_shapelet,
                                        length_increment=length_increment,
                                        position_increment=position_increment,
                                        n_neighbors=n_neighbors,
                                        n_jobs=num_cpu,
                                        log_dir=dir_shapelet)
    r_row = (name, acc, len(best_shapelets), min_shapelet_length, max_shapelet_length)
    file_result.write(output.row2markdown(r_row) + '\n')
    result.append(r_row)


for item in result:
    print(item)




