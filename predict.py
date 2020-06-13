import argparse
from util import process_image, load_class_name_map

import torch
from torch import nn
from torchvision import models

from pathlib import Path
from workspace_utils import active_session

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained=True) if checkpoint['pretrained_type'] == 'vgg19' else None
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = nn.Sequential(checkpoint['classifier'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def predict(image_path, model, topk=5, device='cpu'):
    '''Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print(f'Inferring using {device}...')
    model.to(device)

    # Retrieve file and covert to tensor
    input_im = torch.tensor(process_image(image_path)).type(torch.FloatTensor)

    # Retrieve label from input image path
    label_class = Path(image_path).parts[2]

    # Switch class mapping for convenience
    idx_to_class = {d:k for k,d in model.class_to_idx.items()}

    model.eval()

    with active_session():
        with torch.no_grad():
            input_im = input_im.to(device)
            logps = model(input_im.unsqueeze_(0))
            ps = torch.exp(logps)

            # Get top probabilities, indices
            top_p, top_idx = ps.topk(topk, dim=1)

            # Convert top indices to classes
            top_class = [idx_to_class[k] for k in top_idx.tolist()[0]]

            model.train()    
            input_im = input_im.squeeze().cpu()

            # Return top probabilities, corresponding classes, input image tensor, and correct label class
            return top_p.tolist()[0], top_class, input_im, label_class

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', action="store")
    parser.add_argument('checkpoint_path', action="store")
    parser.add_argument('--top_k', action="store",
                        type=int, default='1')
    parser.add_argument('--category_names', action="store",
                        dest="category_names")
    parser.add_argument('--gpu', action="store_true",
                        default=False)

    args = parser.parse_args()

    # Get class names from map
    class_to_name = load_class_name_map(args.category_names) if args.category_names else None
    
    # Get model and infer
    model = load_checkpoint(args.checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    probs, top_classes, img_tensor, label_class = predict(args.image_path, model, args.top_k, device)
    
    # Print output. Indicates that name is unspecified if no name mapping is provided
    print()
    
    if class_to_name:
        actual_class_name = class_to_name[label_class].title()
        inferred_class_name = class_to_name[top_classes[0]].title()
    else:
        actual_class_name = inferred_class_name = 'Unspecified name'
        
    print(f'Actual class: {actual_class_name} (#{label_class})')
    print(f'Inferred class: {inferred_class_name} (#{top_classes[0]})')
    result = 'Correct' if label_class == top_classes[0] else 'Incorrect'
    print(f'Result: {result}')
    print()
    print('Top inferred classes:')
    for i in range(len(probs)):
        if class_to_name:
            inferred_class_name = class_to_name[top_classes[i]].title()
        else:
            inferred_class_name = 'Unspecified name'
        print(f'{i+1}. {(probs[i]*100):.7f}% - {inferred_class_name} (#{top_classes[i]})')
    print()
   
if __name__ == '__main__':
    main()
    
    

