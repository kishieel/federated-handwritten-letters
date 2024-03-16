### Start server

```shell
python server.py \
  --server-address=0.0.0.0:8080 \
  --num-rounds=6
```

### Start client

```shell
python client.py \
  --server-address=127.0.0.1:8080 \
  --data-dir=./data/client-1
```

### Predict single image

```shell
python predict.py \
    --image-path=./data/client-2/g/1691660116597.png \
    --model-path=./models/model-round-6.keras
```

### Draw confusion matrix

```shell
python confusion.py 
    --model-path=./models/model-round-6.keras
    --data-dir=./data/client-2
```