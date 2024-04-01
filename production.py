import torch
from torchvision import transforms
from PIL import Image
from generator_model import Generator

def load_model(model_path, device):
    model = Generator(img_channels=3, num_residuals=9).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    print('starting to load model')
    # Check if the checkpoint contains a 'state_dict' key
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print('loaded model')
    return model

def transform_single_image(image_path, model, device, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        transformed_image = model(image)

    transformed_image = (transformed_image * 0.5 + 0.5).squeeze(0)  # rescale to [0, 1]
    transformed_image = transforms.ToPILImage()(transformed_image)
    print('transformed image')
    return transformed_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "genh.pth.tar" 
model = load_model(model_path, device)

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_path = "./data/train/damaged/IMG_6059.jpg"  # Update this path
transformed_image = transform_single_image(image_path, model, device, transform)
transformed_image.show()  # or save it using transformed_image.save("output_path.jpg")