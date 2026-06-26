import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from model import resnet20, resnet32, resnet56
from utils import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, set_seed

MODEL_DIR = Path('models')
DEFAULT_CHECKPOINT = MODEL_DIR / 'best.pt'

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_model(checkpoint_path, model_name, device):
    model_fns = {'resnet20': resnet20, 'resnet32': resnet32, 'resnet56': resnet56}
    model = model_fns[model_name]()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def predict_image(image, model, device):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    top3_probs, top3_indices = probs.topk(3)

    results = {}
    for i in range(3):
        class_name = CLASS_NAMES[top3_indices[i].item()]
        confidence = top3_probs[i].item() * 100
        results[class_name] = confidence

    return results


def cli_inference(image_path, model_name, checkpoint):
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        print('Run "python train.py" first to train a model.')
        sys.exit(1)

    model = load_model(checkpoint_path, model_name, device)
    print(f'Model: {model_name} | Checkpoint: {checkpoint_path}')
    print(f'Device: {device}')

    if not Path(image_path).exists():
        print(f'Error: Image not found at {image_path}')
        sys.exit(1)

    results = predict_image(image_path, model, device)

    print(f'\nImage: {image_path}')
    print('-' * 40)
    for i, (cls_name, conf) in enumerate(results.items()):
        marker = ' *' if i == 0 else ''
        print(f'  {i+1}. {cls_name:<12s} {conf:.2f}%{marker}')
    print('-' * 40)


def gradio_interface(image, model_name, checkpoint):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        return {f'Error: Checkpoint not found at {checkpoint}': 0}

    model = load_model(checkpoint_path, model_name, device)
    results = predict_image(image, model, device)

    return {f'{k} ({v:.1f}%)': v / 100 for k, v in results.items()}


def launch_gradio(model_name, checkpoint):
    demo = gr.Interface(
        fn=lambda img: gradio_interface(img, model_name, checkpoint),
        inputs=gr.Image(type='pil', label='Upload Image'),
        outputs=gr.Label(num_top_classes=3, label='Prediction'),
        title='CIFAR-10 Image Classifier',
        description='Upload an image to classify it into one of 10 categories: '
                     'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.',
        examples=[],
        theme='default',
    )
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('image', nargs='?', help='Path to image file (CLI mode)')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet32', 'resnet56'])
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    set_seed(42)

    if args.image:
        cli_inference(args.image, args.model, args.checkpoint)
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f'Warning: Checkpoint not found at {checkpoint_path}')
            print('Run "python train.py" first to train a model.')
            print('Launching Gradio anyway (predictions will fail until model is trained).')
        launch_gradio(args.model, args.checkpoint)


if __name__ == '__main__':
    main()
