import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from model import resnet20
from utils import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, set_seed

MODEL_DIR = Path('models')
DEFAULT_CHECKPOINT = MODEL_DIR / 'best.pt'

UNKNOWN_CLASS = 'unknown/not-cifar10'
CONFIDENCE_THRESHOLD = 15.0
MULTI_OBJECT_GAP = 15.0

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


def predict_with_analysis(probs):
    top3_probs, top3_indices = probs.topk(3)
    top1_conf = top3_probs[0].item() * 100
    top2_conf = top3_probs[1].item() * 100

    results = []
    for i in range(3):
        class_name = CLASS_NAMES[top3_indices[i].item()]
        confidence = top3_probs[i].item() * 100
        results.append((class_name, confidence))

    warnings = []

    if top1_conf < CONFIDENCE_THRESHOLD:
        warnings.append(f'Low confidence: Image may not be a CIFAR-10 object (max {top1_conf:.1f}%)')

    gap = top1_conf - top2_conf
    if gap < MULTI_OBJECT_GAP and top1_conf >= CONFIDENCE_THRESHOLD:
        warnings.append(f'Multiple objects detected: Top-2 predictions are close ({gap:.1f}% gap)')

    return results, warnings


def predict_cli(image_path, checkpoint_path, device):
    if not Path(image_path).exists():
        print(f'Error: Image not found: {image_path}')
        return

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = resnet20()(img_tensor) if device.type == 'cpu' else None
        model = load_model(checkpoint_path, device)
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    results, warnings = predict_with_analysis(probs)

    print(f'\nImage: {image_path}')
    print(f'Size: {image.size}')
    print('-' * 45)
    for i, (cls_name, conf) in enumerate(results):
        marker = ' <<< TOP' if i == 0 else ''
        print(f'  {i + 1}. {cls_name:<12s} {conf:6.2f}%{marker}')
    print('-' * 45)

    if warnings:
        print()
        for w in warnings:
            print(f'  [!] {w}')


def gradio_predict(image):
    if image is None:
        return {UNKNOWN_CLASS: 1.0}, 'Please upload an image'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = DEFAULT_CHECKPOINT

    if not checkpoint_path.exists():
        return {f'Error: No model found. Run train.py first.': 1.0}, 'Model not trained'

    model = load_model(checkpoint_path, device)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    results, warnings = predict_with_analysis(probs)

    top1_conf = results[0][1]
    if top1_conf < CONFIDENCE_THRESHOLD:
        label_dict = {UNKNOWN_CLASS: 1.0}
    else:
        label_dict = {}
        for class_name, conf in results:
            label_dict[f'{class_name} ({conf:.1f}%)'] = float(conf / 100)

    warning_text = '\n'.join(warnings) if warnings else ''

    return label_dict, warning_text


def launch_gradio():
    demo = gr.Blocks(title='CIFAR-10 Image Classifier')
    with demo:
        gr.Markdown('# CIFAR-10 Image Classifier')
        gr.Markdown('Upload an image to classify — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.')

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='pil', label='Upload Image')
                submit_btn = gr.Button('Classify', variant='primary')
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=3, label='Predictions')
                warning_output = gr.Textbox(label='Notes', placeholder='No warnings')

        submit_btn.click(
            fn=gradio_predict,
            inputs=image_input,
            outputs=[label_output, warning_output]
        )

        gr.Markdown('---')
        gr.Markdown(f'''
        **About:** CIFAR-10 classifier using ResNet-20 (92.77% test accuracy).
        Images are auto-resized to 32x32 before classification.

        **Thresholds:**
        - Confidence < {CONFIDENCE_THRESHOLD}% → flagged as "unknown"
        - Top-2 gap < {MULTI_OBJECT_GAP}% → multiple objects warning
        ''')

    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('image', nargs='?', help='Path to image file (CLI mode)')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    set_seed(42)

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
