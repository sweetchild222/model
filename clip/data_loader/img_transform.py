from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def img_transform(img_resolution):    

    return Compose([
        Resize(img_resolution, interpolation=Image.BICUBIC),
        CenterCrop(img_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)) #mean, std
    ])
