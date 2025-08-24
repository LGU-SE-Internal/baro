# baro

```bash
sudo juicefs mount redis://10.10.10.38:6379/1 /mnt/jfs -d --cache-size=1024

mkdir data
cd data
ln -s /mnt/jfs/rcabench_dataset ./
ln -s /mnt/jfs/rcabench-platform-v2 ./

./main.py eval single baro rcabench ts3-ts-route-plan-service-request-delay-59s2q4

mkdir temp
ALGORITHM=baro INPUT_PATH=data/rcabench_dataset/ts3-ts-route-plan-service-request-delay-59s2q4 OUTPUT_PATH=temp uv run python run_exp.py
```

```sh
export RCABENCH_BASE_URL=http://10.10.10.220:32080
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
sudo -E .venv/bin/python run.py batch-test


docker build -t 10.10.10.240/library/rca-algo-baro:820c2d7 .
docker push 10.10.10.240/library/rca-algo-baro:820c2d7
rca upload-algorithm-harbor ./
```