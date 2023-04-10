# LLM Racing
A simple set of command line tools for evaluating language models generation speed.

## Replication
This library uses deep speed as a backend for inference. This can be tricky to
setup so I recommend using the provided docker image for replicating these results.

```
git clone https://github.com/levmckinney/llm-racing.git
cd llm-racing
```
Then pull the docker image and run it. Note you will need the [nvidia container took](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed for this to work. The default image is built for A100 and V100. You will need to rebuild the docker file for other compute architectures.
```
docker run -it --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -v .:/workspace/ \
    ghcr.io/levmckinney/llm-racing:latest
```
Then run the experiments and produce the plots.
```
$ pip install -e .
$ bash run_experiment.sh
$ python3 -m llm_racing plot
```
