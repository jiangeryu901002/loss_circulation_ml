# loss_circulation_ml
Codes for the lost circulation prediction

## Chronos-2 environment setup
`conda create -n chronos python=3.12`

`conda activate chronos`

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

`pip install chronos-forecasting>=2.0 pandas[pyarrow] matplotlib`

## Run Chronos-2 
`python Chronos2_cov.py`