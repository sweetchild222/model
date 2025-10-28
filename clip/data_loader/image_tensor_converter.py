from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


class ImageTensorConverter:

    def __init__(self, tensor_resolution):
        
        self.tensor_resolution = tensor_resolution
        self.compose = self.composer(self.tensor_resolution)
    
    def composer(self, resolution):

        def convert_rgb(image):
            return image.convert("RGB")

        return Compose([Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(resolution),
                convert_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    

    def covert(self, image_path):

        image_rgb = Image.open(image_path)

        return self.compose(image_rgb)