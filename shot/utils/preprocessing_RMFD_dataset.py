import pickle

import numpy as np
from torchvision.datasets import ImageFolder

face_dataset = ImageFolder('../../data/RMFD/AFDB_face_dataset')
mask_dataset = ImageFolder('../../data/RMFD/AFDB_masked_face_dataset')

print(face_dataset.classes)
print(mask_dataset.classes)

face_classes = np.asarray(face_dataset.classes)
mask_classes = np.asarray(mask_dataset.classes)

same_classes = list()

for face_class in face_classes:
    if face_class in mask_classes:
        same_classes.append(face_class)

with open('../../data/same_classes_list.pkl', 'wb') as f:
    pickle.dump(same_classes, f)
