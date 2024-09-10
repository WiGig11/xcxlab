# THis is a readme file for the work titiled:
《Eyeglass Reflection Removal with Joint Learning of Reflection Elimination and Content Inpainting》

## overall structure
### A shared Encoder
however, in real use, this encoder seems to be used only by the inpainting branch

### Three branches
#### inpainting branch
This branch uses a encoder and decoder of its own
The result is passed through a memory blcok .the memory block is initliazed as a matrix and as a member of the class so that after initlization, no more space is needed.
Forward contains two function , read and update , the function returns the read result(updated query ) and the updated memory is restored with
```python
self.memory = updated_memory
```
encoder - memory - decoder

#### detection branch
This branch shares a encoder with elinination branch and uses a decoder of its own
the result is supervised using focal loss, using $I_{input}$ and $I_{gt}$ to make a mask indicting the reflection area and use the focal loss to optimize the encoder and decoder

#### elimination branch
This branch shares a encoder with detection branch and uses a decoder of its own 
and passes a resblock ( contains in the resblock )

### eye flow net
This is a bit strange because the decoder should use a transposed conv rather than conv


## loss
### rec loss
I used mse loss (L2 loss), however in open-sources codes, a more advanced weights loss is used

### precetion loss
VGG LOSS

### focal loss
discuseed above

### weight loss
weight loss to regulate weight map from RFM

### eye loss
flow should make eye align

### flow loss
landmark loss and gradient loss from four directions
