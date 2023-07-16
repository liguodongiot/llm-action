import numpy as np
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

# preprocessing function
def rn50_preprocess(img_path="img1.jpg"):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).numpy()

transformed_img = rn50_preprocess()

np.set_printoptions(threshold=np.inf)

#print(transformed_img)


# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=3)

#outputs = httpclient.InferRequestedOutput("output__0", binary_data=True)

# Querying the server
results = client.infer(model_name="resnet50", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('output__0')
print(inference_output[:5])

print(inference_output.shape)
