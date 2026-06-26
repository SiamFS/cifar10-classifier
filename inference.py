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
    MODEL_DIR, INFERENCE_CONFIDENCE_THRESHOLD, GRADIO_SERVER_PORT, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CURRENT_TEST_ACCURACY,
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
    all_probs = (probs * 100).cpu().numpy()

    detected = []
    for i in range(len(CLASS_NAMES)):
        conf = float(all_probs[i])
        if conf >= INFERENCE_CONFIDENCE_THRESHOLD:
            detected.append((CLASS_NAMES[i], conf))
    detected.sort(key=lambda x: -x[1])

    results = detected[:3] if len(detected) >= 3 else detected

    warnings = []
    if len(detected) == 0:
        warnings.append(f'Nothing detected: all classes below {INFERENCE_CONFIDENCE_THRESHOLD}%. Image may be out of domain.')
    elif len(detected) > 1:
        names = [d[0] for d in detected]
        warnings.append(f'Multiple objects detected: {", ".join(names)}')

    return results, warnings, detected


def predict_cli(image_path, checkpoint_path, device):
    if not Path(image_path).exists():
        print(f'Error: Image not found: {image_path}')
        return
    image = Image.open(image_path).convert('RGB')
    model = load_model(checkpoint_path, device)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img_tensor), dim=1)[0]
    results, warnings, detected = predict_with_analysis(probs)
    print(f'\nImage: {image_path} | Size: {image.size}')
    print('-' * 45)
    for i, (cls_name, conf) in enumerate(results):
        marker = ' <<< TOP' if i == 0 else ''
        print(f'  {i + 1}. {cls_name:<12s} {conf:6.2f}%{marker}')
    if len(detected) > 3:
        print(f'  ... and {len(detected) - 3} more above threshold')
    print('-' * 45)
    if warnings:
        for w in warnings:
            print(f'  [!] {w}')
    else:
        print(f'  All classes below {INFERENCE_CONFIDENCE_THRESHOLD}% — may be out of domain')


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
    results, warnings, detected = predict_with_analysis(probs)

    if len(detected) == 0:
        label_dict = {'No CIFAR-10 objects detected': 1.0}
    else:
        label_dict = {f'{class_name} ({conf:.1f}%)': float(conf / 100) for class_name, conf in detected}

    return label_dict, '\n'.join(warnings) if warnings else ''


def launch_gradio():
    demo = gr.Blocks(title='CIFAR-10 Image Classifier')
    with demo:
        gr.Markdown('# CIFAR-10 Image Classifier')
        gr.Markdown('Classify images into: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.')
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='pil', label='Upload Image', sources=['upload'])
                submit_btn = gr.Button('Classify', variant='primary')
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=3, label='Predictions')
                warning_output = gr.Textbox(label='Notes', placeholder='No warnings')
        submit_btn.click(fn=gradio_predict, inputs=image_input, outputs=[label_output, warning_output])
        gr.Markdown(f'''
        **How it works:** Upload a photo of a single object — like a cat, a dog, or a car — and the model will identify it.
        **Limitations:** Images containing multiple objects (e.g., a cat and a bird together) cannot be classified correctly. Multi-object support is under development.
        **Note:** Confidence below {INFERENCE_CONFIDENCE_THRESHOLD}% means the image may not contain any of the 10 supported categories. Images are auto-resized to 32x32.
        ''')
    print(f'\nOpen http://localhost:{GRADIO_SERVER_PORT} in your browser')
    print('Upload any image to classify it.\n')
    demo.launch(server_port=GRADIO_SERVER_PORT, share=False)


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
