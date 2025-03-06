Phy_STLNet

------------------------------------
Requirements:
- torch: PyTorch 1.4
- tqdm: Progress bar

------------------------------------
Usage:
Run the code for a specific dataset

python main.py --data (air/cont) [--lr 0.001 --lambdao 10 --epochs 10]

python main.py --timeunites 24 --data airpde1 --cell_type pde --lr 0.1 --lambdao 1 --epochs 10

python main.py --timeunites 24 --data airpde2 --cell_type pde --lr 0.1 --lambdao 1 --epochs 10


------------------------------------

The old command was not supportable in the new model. If you want to run the previous baseline, please check https://github.com/meiyima/STLnet

------------------------------------

Script for benchmarking
- Air Quality Data
```
./runair.sh
```
- Synthesized data
```
./rungen.sh
```

-------------------------------------
Air Quality Data:
We use public air quality data from data source: https://github.com/yoshall/GeoMAN.
