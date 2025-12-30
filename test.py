from keras.metrics import Accuracy, SparseCategoricalAccuracy
import numpy as np

y = np.array([5., 4.])

y_pred = np.array([[0.58632016, -0.16838713,  0.3608763,  -0.01111207,  0.21863163, -0.12012183,
  -0.21094704, -0.23382297,  1.1478745,  -0.05678384],
 [-0.05143715,  0.03590139,  0.9020542,   0.84351933, -0.20218468, -0.01276064,
   0.03080075, -0.33579516, -0.19136038,  0.46369603]])

print(f"SparseCategoricalAccuracy: {SparseCategoricalAccuracy()(y, y_pred)}")
