# Load custom dataset in MXNet

We download the Flickr Material Dataset (FMD), split it into training and validation sets (ratio specificable), and create resized (specifiable) dataset. MXNet recommends the [RecordIO](http://mxnet.io/architecture/note_data_loading.html) format, which
concatenates multiple examples into seekable binary files for better read
efficiency. This can be done using code in the MXNet Python package.

## Download and extract dataset

   ```bash
  wget https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
  unzip FMD.zip
  ```
You should now see two folders `image` and `mask`. For the purpose of this tutorial, only `image` is relevant.


The FMD dataset has the following 10 classes:
- Fabric
- Foliage
- Glass
- Leather 
- Metal 
- Paper 
- Plastic 
- Stone 
- Water 
- Wood

Each class has 100 sample images, and images belonging to the same class are placed in
the same directory. All these class directories are then in the same root
`image` directory. 

Our goal is to generate two files, `fmd_train.rec` for
training and `fmd_val.rec` for validation, with a 70-30 split between these two sets. To do this, we first need to prepare "list" files.

## Prepare list files

We first prepare two `.lst` files, which consist of the labels and image paths can be used for generating `rec` files.

For this, we need to the `im2rec.py` scripts in the MXNet package. A variable `m` is first set to denote the directory containing the MXNet package (refactor as per your system):

  ```bash
  m="/Users/slakshay/anaconda3/lib/python3.6/site-packages/mxnet"
  ```

We now proceed with creating the two `.lst` files which consist of the labels and image paths can be used for generating rec files. Note the value `0.7` specifiying the train-val split. Parameter `image` is the name of the extracted directory containing the relevant data, and `fmd` specifies the prefix we attach to our training and validation data.

```bash
python $m/tools/im2rec.py --list --recursive --train-ratio 0.7 fmd image
```

The script takes the list of names of all of the images, shuffles them, then separates them into two lists: a training filename list and a testing filename list.

New files `fmd_train.lst` and `fmd_val.lst` should now be visible. The terminal should show output class labels and the number associated with each class.

## Prepare RecordIO files

The `.rec` files can now be prepared. We resize the images such that the short edge is at least 240px and save them with 95/100 quality. Also, 16 threads are used to accelerate the packing.

```bash
python $m/tools/im2rec.py --resize 240 --quality 95 --num-thread 16 fmd image
```

New files `fmd_train.rec`, `fmd_val.rec`, `fmd_train.idx`, and `fmd_val.idx` should now be visible. The terminal should display values for `time` and `count`.

## References

- https://github.com/apache/incubator-mxnet/tree/master/example/image-classification
- https://mxnet.incubator.apache.org/faq/recordio.html
- http://mxnet.incubator.apache.org/faq/finetune.html
