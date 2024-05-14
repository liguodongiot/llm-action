

- https://pytorch.org/docs/stable/elastic/run.html







```
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```



For multi-node training you need to specify:

--rdzv-id: A unique job id (shared by all nodes participating in the job)

--rdzv-backend: An implementation of torch.distributed.elastic.rendezvous.RendezvousHandler

--rdzv-endpoint: The endpoint where the rendezvous backend is running; usually in form host:port.

Currently c10d (recommended), etcd-v2, and etcd (legacy) rendezvous backends are supported out of the box. To use etcd-v2 or etcd, setup an etcd server with the v2 api enabled (e.g. --enable-v2).



