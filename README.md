This is a C++ implementation of the SNN manifesto design by Eugene Izhikevich. 
His original files are included, main.c, readme and license. 
To use this, extract the zip file, build through Linux terminal, and run with the commands in snncommands.txt.
This adds the ability to carry over the same model to different tests, pass a sample inference, and choose varying types of files as the input.


**ORIGINAL**<br/>
**SNN Transformer** <br />
Eugene Izhikevich (2025) Spiking Manifesto, https://arxiv.org/pdf/2512.11843 <br />
Hardware patent pending 

**Data for training**

Use any sufficiently large text file. Provide filename in line 546


**Compile and Run**

gcc -O3 -o main main.c -lm <br />
./main

Validation loss values during training are saved in the FILE_NAME in line 16

**Deviations from the model in the paper**

1. Positional encoders PE are different for different look-up tables; see void cache_PE_index()

2. Positional encoders do not use anchors neurons; indexes are formed using element-wise comparisons with 0. Hence, the number of comparisons is POSITIONAL_DIM (parameter p in the paper)

3. Anchor neurons for z_i and z_j are the same, so their indexes are re-used by the V-index cache

4. The learning of token_embedder is disabled. It does not improve performance
