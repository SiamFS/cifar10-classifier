import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from model import resnet20
from config import (
    CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, RANDOM_SEED,
    MODEL_DIR, INFERENCE_CONFIDENCE_THRESHOLD, INFERENCE_MULTI_OBJECT_GAP,
    GRADIO_SERVER_PORT,
)
from utils import set_seed

DEFAULT_CHECKPOINT = MODEL_DIR / 'best.pt'
_model_cache = None
_device = None

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_model(checkpoint_path, device):
    model = resnet20()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_model():
    global _model_cache, _device
    if _model_cache is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if DEFAULT_CHECKPOINT.exists():
            _model_cache = load_model(DEFAULT_CHECKPOINT, _device)
        else:
            raise FileNotFoundError(f'Model not found at {DEFAULT_CHECKPOINT}. Run train.py first.')
    return _model_cache, _device


def predict_with_analysis(probs):
    top3_probs, top3_indices = probs.topk(3)
    top1_conf = top3_probs[0].item() * 100
    top2_conf = top3_probs[1].item() * 100
    results = [(CLASS_NAMES[top3_indices[i].item()], top3_probs[i].item() * 100) for i in range(3)]
    warnings = []
    if top1_conf < INFERENCE_CONFIDENCE_THRESHOLD:
        warnings.append(f'Low confidence: Image may not be a CIFAR-10 object (max {top1_conf:.1f}%)')
    gap = top1_conf - top2_conf
    if gap < INFERENCE_MULTI_OBJECT_GAP and top1_conf >= INFERENCE_CONFIDENCE_THRESHOLD:
        warnings.append(f'Multiple objects detected: Top-2 predictions are close ({gap:.1f}% gap)')
    return results, warnings


def predict_cli(image_path, checkpoint_path, device):
    if not Path(image_path).exists():
        print(f'Error: Image not found: {image_path}')
        return
    image = Image.open(image_path).convert('RGB')
    model = load_model(checkpoint_path, device)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img_tensor), dim=1)[0]
    results, warnings = predict_with_analysis(probs)
    print(f'\nImage: {image_path} | Size: {image.size}')
    print('-' * 45)
    for i, (cls_name, conf) in enumerate(results):
        marker = ' <<< TOP' if i == 0 else ''
        print(f'  {i + 1}. {cls_name:<12s} {conf:6.2f}%{marker}')
    print('-' * 45)
    if warnings:
        for w in warnings:
            print(f'  [!] {w}')


def gradio_predict(image):
    if image is None:
        return {'No image uploaded': 1.0}, ''
    try:
        model, device = get_model()
    except FileNotFoundError as e:
        return {str(e): 1.0}, 'Model not trained'
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img_tensor), dim=1)[0]
    results, warnings = predict_with_analysis(probs)
    top1_conf = results[0][1]
    if top1_conf < INFERENCE_CONFIDENCE_THRESHOLD:
        label_dict = {'unknown/not-cifar10': 1.0}
    else:
        label_dict = {f'{class_name} ({conf:.1f}%)': float(conf / 100) for class_name, conf in results}
    return label_dict, '\n'.join(warnings) if warnings else ''


def launch_gradio():
    demo = gr.Blocks(title='CIFAR-10 Image Classifier')
    with demo:
        gr.Markdown('# CIFAR-10 Image Classifier')
        gr.Markdown('Classify images into: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.')
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='pil', label='Upload Image')
                submit_btn = gr.Button('Classify', variant='primary')
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=3, label='Predictions')
                warning_output = gr.Textbox(label='Notes', placeholder='No warnings')
        submit_btn.click(fn=gradio_predict, inputs=image_input, outputs=[label_output, warning_output])
        gr.Markdown(f'''
        **About:** ResNet-20 (92.88% test accuracy) | Auto-resizes to 32x32.
        Confidence < {INFERENCE_CONFIDENCE_THRESHOLD}% = flagged as unknown.
        Top-2 gap < {INFERENCE_MULTI_OBJECT_GAP}% = multiple objects warning.
        ''')
    print(f'\nOpen http://localhost:{GRADIO_SERVER_PORT} in your browser')
    print('Upload any image to classify it.\n')
    demo.launch(server_name='127.0.0.1', server_port=GRADIO_SERVER_PORT, share=False)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('image', nargs='?', help='Path to image file (CLI mode)')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    set_seed(RANDOM_SEED)
    if args.image:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f'Error: Checkpoint not found at {checkpoint_path}')
            print('Run "python train.py" first to train the model.')
            return
        predict_cli(args.image, checkpoint_path, device)
    else:
        launch_gradio()


if __name__ == '__main__':
    main()
