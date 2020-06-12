import argparse
from PIL import Image

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
print(f'image_path: {args.image_path} {type(args.image_path)}')
print(f'checkpoint_path: {args.checkpoint_path} {type(args.checkpoint_path)}')
print(f'top_k: {args.top_k} {type(args.top_k)}')
print(f'category_names: {args.category_names} {type(args.category_names)}')
print(f'gpu: {args.gpu} {type(args.gpu)}')

def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Resize and crop
    width, height = image.size
    old_short, old_long = min([width, height]), max([width, height])
    new_short = 256
    new_long = int(new_short*old_long/old_short)
    new_size = [new_short, new_long] if old_short == width else [new_long, new_short]
    image = image.resize(new_size)
    
    center_size = 224
    width, height = image.size
    left = (width - center_size)/2
    right = (width + center_size)/2
    top = (height - center_size)/2
    bottom = (height + center_size)/2
    image = image.crop((left, top, right, bottom))
    width, height = image.size
    
    # Normalize
    np_image = np.array(image)/255.0
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]
    
    # Transpose
    np_image = np_image.transpose(2, 0, 1)
    return np_image

model = load_checkpoint(checkpoint_path)

def predict(image_path, model, topk=5):
    '''Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Retrieve file and covert to tensor
    f = Image.open(image_path)
    input_im = torch.tensor(process_image(f)).type(torch.FloatTensor)
    
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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained=True) if checkpoint['pretrained_type'] == 'vgg19' else None
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = nn.Sequential(checkpoint['classifier'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
    
probs, classes, img_tensor, label_class = predict(image_path, model, top_k)
