# VQA config

We provide several vqa model with/without CBN.

The configuration file is divided into different parts:
 - Network
   - image processing
   - network architecture
 - Training features
 - Misc (seed etc.)

The keyword "model" refers to the VQA network:
```
 "model": {

    "name" : "MRN network (small) with with 2 glimpses", # Basic dfescription

    "image":
    {
      "image_input": str,                # Image input raw/conv/fc8/fc7 cf below for more information
      "dim": [int],                      # Dimension of your image input (width/height/featres)
      "normalize": bool,                 # Nornalize the image features

      "resnet_version": int,             # 50/101/152 Choose the resnet you need. Must be in the ckpt must be in the data folder
      "finetune" : ["str"]               # Finetune specific parts of the ResNet (raw only)

      "attention" : {                    # Configure your attention mechanism (Automatically ignore if you use fc7/fc8)
        "mode": str,                     # none/classic/glimpse = Mean / classic attention / glimpse attention
        "no_attention_mlp": int,         # Size of the attention embedding space
        "no_glimpses": int               # Number of glimpse (1=classic attention)
      },

      "cbn": {
        "use_cbn": bool,                 # Use Conditional BatchNormalization in the ResNet
        "cbn_embedding_size": int        # Number of units used to compute CBN parameters for each feature map
        "excluded_scope_names": ["str"], # Do not apply CBN to specific feature map. Wildcard caracteres skip all feature maps
      }

    },

    "glove": bool,                 # Use GLOVE

    "word_embedding_dim": int,     # Dimension of word embeddings
    "no_hidden_LSTM": int,         # Dimension of the LSTM (no of hidden units)
    "no_LSTM_cell": int,           # Number of stacked LSTM

    "no_question_mlp": int,        # Dimension of the LSTM projection
    "no_image_mlp": int,           # Dimension of the Image projection (Note they must be both equals)

    "no_hidden_final_mlp": int,    # No of units before the final softmax
    "dropout_keep_prob": float,    # Dropout ratio (1 = no dropout)

    "loss": str,                   # soft/hard use trucated probability distribution or majority vote
    "activation": str,             # Activation units

    }
```

One may use the input image he wants:
 - raw: images with pixels. A pretrained ResNet will be loaded
 - conv: pre-computed image features from ResNet or others
 - fc8/fc7: VGG image features

Please refer to the GuessWhat README if you need to precompute those image features

"finetune" and "excluded_scope_names" parameters are going to match a string inside the name of the variable.
For instance, "block4" will impact all the variables that have block4 in their name.


The "optimizer" key refers to the training hyperparameters:

```
  "optimizer": {
    "no_epoch": int,            # the number of traiing epoches
    "learning_rate": float,     # SGD initial learning rate
    "batch_size": int,          # training batch size
    "clip_val": int             # gradient clip to avoid RNN gradient explosion
  },
 ```

Other parameters can be set such as:

```
  "dico_name" : "str",      # name of the dico file in the data directectory
  "glove_name" : "str",     # name of the glove file in the data directectory

  "merge_dataset": false,   # fuse training and validation sets at training time
  "seed": int               # define the training seed; -1 -> random seed
 ```

 Note that we put the dico/glove file in the config file on purpose (instead of command line argument).
 Indeed, those dictionary are key parameters of the training (no of words, val+train etc.)