# AI_Edge_segmentation

This is SIGNATE 4th AI Edge Contest results that are mainly quantized segmentation model working on hardware as FPGA(ultra96v2).

Simply, show this repositry characteristic parts.

# segmentation result by quantized model on FPGA

input image & result image


<img src="https://user-images.githubusercontent.com/48679574/103159027-ad09be80-4807-11eb-9eda-2d8daf13be6a.jpg" width="400px"><img src="https://user-images.githubusercontent.com/48679574/103159033-bbf07100-4807-11eb-8c01-0fd4ce7bfc7b.png" width="400px">


<img src="https://user-images.githubusercontent.com/48679574/146494490-12201c8c-2014-46bc-96b3-e895607604be.jpg" width="400px"><img src="https://user-images.githubusercontent.com/48679574/146494496-8d225041-7a04-4457-a7ac-3d75c0dd4cc8.png" width="400px">



# this AI eadge characteristic point

## Overall flow to run on ultra96v2(AI edge)

model libarary version is 
```
keras==2.2.4
tensorflow==1.13.1
```

<img src="https://user-images.githubusercontent.com/48679574/146493987-c11c3190-7fa3-4a82-978b-bd8a34d27f80.png" width="500px">


## Hardware platform 

<img src="https://user-images.githubusercontent.com/48679574/146493975-44a491b9-fec0-45f2-980d-dded6850a2fd.png" width="500px">



## Parallel processing by C++ Multiple thread

<img src="https://user-images.githubusercontent.com/48679574/146493991-75911008-9f82-4226-8572-507d80a98adc.png" width="500px">


## this logic is summarized in my blog 
https://trafalbad.hatenadiary.jp/entry/2020/12/21/123343


