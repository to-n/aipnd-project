#/bin/bash

python predict.py flowers/test/23/image_03390.jpg checkpoint/checkpoint.pth --top_k=5 --category_names=cat_to_name.json --gpu
python predict.py flowers/test/23/image_03390.jpg checkpoint/checkpoint.pth --top_k=5 --gpu
python predict.py flowers/test/87/image_05466.jpg checkpoint/checkpoint.pth --top_k=5 --category_names=cat_to_name.json --gpu
python predict.py flowers/test/87/image_05466.jpg checkpoint/checkpoint.pth --top_k=5 --gpu