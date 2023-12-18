# CNN for age estimation

## Reference
Reference the following publication: <br/>
Coming soon.

## model weight
Download model weight:<br/>
https://www.dropbox.com/scl/fi/b3wkkpd54s0f8ojophm0e/2216_weight.h5?rlkey=ajdu14emfm3ekhtij16lrwz9l&dl=0


## Instructions
* Code to apply the model
   
<pre>
```python
import pydicom
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from skimage.transform import resize

def get_my_model(learning_rate = 1e-4, d = 0.25):
    # ...
    
def preprocess_dicom_image(file_path):
    return resize(pydicom.read_file(file_path).pixel_array, (256, 256)).reshape(1, 256, 256, 1) 

def estimate_age(model, input_image):
    return model.predict(input_image)[0, 0]

if __name__ == "__main__":
    model = get_my_model()
    model.load_weights('2216_weight.h5')

    dicom_image_path = '...'
    print(f"Predicted Age: {estimate_age(model, preprocess_dicom_image(dicom_image_path))} years")
```
</pre>
