## Implementation of the paper [Domain-Adaptive Multibranch Networks](https://infoscience.epfl.ch/record/273445) using Pytorch

Currently only LeNet based Multibranch Network has been implemented

### General Architecture
![](https://github.com/PanPapag/DAMNets/blob/master/images/damnet.png)

### Multibranch LeNet
![](https://github.com/PanPapag/DAMNets/blob/master/images/damnet_lenet.png)

### How to use?
1. Open the terminal
2. Type ```git clone https://github.com/PanPapag/DAMNets.git``` 
   to clone the repository to your local machine
3. Type ```pip install -r requirements.txt```
4. Type ```python main.py --help ``` to view possible options
5. Type ```python main.py ``` to run the app

### TODOs
1. Add more available datasets
2. Add model checkpoint saving option, so as to load a pretrained model and test.
3. Build the other DAMNets (

Feel free to contribute :smiley:

### License
This project is licensed under the MIT License.

MIT © [PanPapag]()
