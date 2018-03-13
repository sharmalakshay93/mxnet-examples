# Load custom dataset in MXNet

We download the Flickr Material Dataset (FMD), split it into training and validation sets (ratio specificable), and create resized (specifiable) dataset. MXNet recommends the [RecordIO](http://mxnet.io/architecture/note_data_loading.html) format, which
concatenates multiple examples into seekable binary files for better read
efficiency. This can be done using code in the MXNet Python package.

## Download and extract dataset

   ```bash
  python train_mnist.py --network mlp
  ```


The FMD dataset has the following 10 classes:
-Fabric
-Foliage
-Glass
-Leather 
-Metal 
-Paper 
-Plastic 
-Stone 
-Water 
-Wood

Each class has 100 sample images, and images belonging to the same class are placed in
the same directory. All these class directories are then in the same root
`image` directory. Our goal is to generate two files, `mydata_train.rec` for
training and `mydata_val.rec` for validation, with a 70-30 split between these two sets.

## Prepare list files

We first prepare two `.lst` files, which consist of the labels and image paths
can be used for generating `rec` files.

```bash
wget https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
unzip FMD.zip```

Then we generate the `.rec` files. We resize the images such that the short edge
is at least 480px and save them with 95/100 quality. We also use 16 threads to
accelerate the packing.

```bash
python tools/im2rec.py --resize 480 --quality 95 --num-thread 16 mydata img_data
```

