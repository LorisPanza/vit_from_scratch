# 👕 ViT: Making Fashion Pixels Trendy Again 👗
Because even AI deserves a fashion upgrade from boring CNNs!
## What's This Fashion-Forward Code About?
This project implements a Vision Transformer (ViT) model inspired by the groundbreaking paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. But instead of using it for something serious, I'm using this computational powerhouse to decide if that pixelated blob is a sandal or an ankle boot. High fashion, am I right?
## 🧠 The Secret Sauce
Remember when everyone said "you need convolutions for images"? Well, this ViT model says "hold my gradient tape" and processes Fashion MNIST using only self-attention. It chops up those stylish 28×28 grayscale masterpieces into 4×4 patches (fashion squares, if you will), and lets transformers do their magic.

## 🚀 How To Make Your Computer Recognize Fashion

1. Clone this repo faster than a fast-fashion brand copies runway designs
2. Run:
   ```shell
    python training.py
   ```
    

4. Watch your GPU sweat through 5 epochs

## Citation
```shell
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
