import os
import pickle
from glob import glob

masked_base_path = '../../data/RMFD/AFDB_masked_face_dataset'
unmasked_base_path = '../../data/RMFD/AFDB_face_dataset'
masked_people_names = [os.path.basename(path) for path in glob(os.path.join(masked_base_path, '*'))]
unmasked_people_names = [os.path.basename(path) for path in glob(os.path.join(unmasked_base_path, '*'))]
masked_people_dataset = {name: glob(os.path.join(masked_base_path, name, '*')) for name in masked_people_names}
unmasked_people_dataset = {name: glob(os.path.join(unmasked_base_path, name, '*')) for name in
                           unmasked_people_names}
masked_people_names = list(filter(lambda x: len(masked_people_dataset[x]) > 0, masked_people_names))
unmasked_people_names = list(filter(lambda x: len(unmasked_people_dataset[x]) != 0, unmasked_people_names))
common_people_names = list(filter(lambda x: (x in masked_people_names) and (x in unmasked_people_names),
                                  list(set(masked_people_names + unmasked_people_names))))

print(len(common_people_names))
with open('../../data/same_classes_list.pkl', 'wb') as f:
    pickle.dump(common_people_names, f)
