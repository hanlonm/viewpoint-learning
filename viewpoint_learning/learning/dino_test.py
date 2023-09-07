from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch 
from PIL import Image 
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA, IncrementalPCA
from pathlib import Path
import pickle
import cv2


resolution = 224
patch_resolution = int(resolution / 14)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
environments = ["00111","00195","00403","00654"]



# image.show()
# url = 'https://www.barcs.org/media/images/chessie.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14.to("cuda")
image_transforms = T.Compose([
    T.Resize((resolution,resolution), interpolation=T.InterpolationMode.BICUBIC),
    #T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])



for environment in environments:

    im_path = Path(f"/local/home/hanlonm/Hierarchical-Localization/datasets/{environment}/mapping/RF_0010.jpeg")
    cv_image = cv2.imread(str(im_path))
    cv2.imwrite(f"/local/home/hanlonm/Pictures/{environment}_unchanged.png", cv_image)
    image = Image.open(im_path)
    file = open("/local/home/hanlonm/mt-matthew/data/pca_3_transform.pkl", 'rb')
    pca: IncrementalPCA = pickle.load(file)
    img = image_transforms(image).unsqueeze(0).to("cuda")
    # plt.imshow(T.ToPILImage()(img.squeeze()))
    # plt.show()
    with torch.no_grad():
        features: torch.Tensor = dinov2_vitb14.forward_features(img)["x_norm_patchtokens"]
        features =features.reshape(1, patch_resolution ,patch_resolution,768)
        features = torch.permute(features, (0,3,1,2))
        features = torch.nn.functional.interpolate(features,(640,480)) # B,N,H,W
        features = torch.permute(features, (0,2,3,1))
        features = features.reshape(1,-1,768)
        print()
        # pca = PCA(n_components=3)
        # pca.fit(features.squeeze().detach().cpu())
        pca_features = pca.transform(features.squeeze().detach().cpu())
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = pca_features * 255


        cv2.imwrite(f"/local/home/hanlonm/Pictures/{environment}_dino.png", pca_features.reshape(640, 480, 3).astype(np.uint8))
        # plt.imshow(pca_features.reshape(640, 480, 3).astype(np.uint8))


# processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
# processor.do_resize=False
# model = ViTModel.from_pretrained('facebook/dino-vitb8')

# inputs = processor(images=image, return_tensors="pt")
# im = T.ToPILImage()(inputs.data["pixel_values"].squeeze())
# plt.imshow(im)
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print()
# plt.show()