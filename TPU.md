64 spot Cloud TPU v5e chips in zone europe-west4-b
32 spot Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v6e chips in zone us-east1-d
32 on-demand Cloud TPU v4 chips in zone us-central2-b
64 spot Cloud TPU v5e chips in zone us-central1-a
64 spot Cloud TPU v6e chips in zone europe-west4-a

https://docs.cloud.google.com/tpu/docs/queued-resources#delete_a_queued_resource_request

```
ssh -o ControlPath=~/.ssh/controlmasters/tpu1a-%r@%h:%p -i ~/.ssh/google_compute_engine muaz@34.13.210.107
tmux capture-pane -t bytelm -p -S -10
```

```
gcloud compute tpus queued-resources list --project raden-tpu --zone us-central2-b
gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-a
gcloud compute tpus queued-resources list --project raden-tpu --zone europe-west4-b
gcloud compute tpus queued-resources list --project raden-tpu --zone us-east1-d

```

```
gcloud compute tpus queued-resources create tpu6 --node-id tpunode6 --project raden-tpu --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base

gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone us-central2-b

gcloud compute tpus queued-resources delete tpu1 --project raden-tpu  --zone us-central2-b --force --async

gcloud compute tpus queued-resources ssh tpu5 --project raden-tpu --zone us-central2-b
```


```
gcloud compute tpus queued-resources create tpu1 --node-id tpunode1 --project raden-tpu --zone europe-west4-a --accelerator-type v6e-1 --runtime-version v2-alpha-tpuv6e --spot

gcloud compute tpus queued-resources describe tpu1 --project raden-tpu --zone europe-west4-a

gcloud compute tpus queued-resources delete tpu1 --project raden-tpu  --zone europe-west4-a --force --async

gcloud compute tpus queued-resources ssh tpu1 --project raden-tpu --zone europe-west4-a
```

```
gcloud compute tpus queued-resources create tpu1 --node-id tpunode1 --project raden-tpu --zone us-east1-d --accelerator-type v6e-1 --runtime-version v2-alpha-tpuv6e --spot# --network-tier=STANDARD

gcloud compute tpus queued-resources describe tpu3--project raden-tpu --zone us-east1-d

gcloud compute tpus queued-resources delete tpu3 --project raden-tpu  --zone us-east1-d --force --async

gcloud compute tpus queued-resources ssh tpu2 --project raden-tpu --zone us-east1-d
```

```
gcloud compute tpus queued-resources create tpu3 --node-id tpunode --project raden-tpu --zone europe-west4-b --accelerator-type v5litepod-1 --runtime-version v2-alpha-tpuv5-lite --spot

gcloud compute tpus queued-resources describe tpu3--project raden-tpu --zone europe-west4-b

gcloud compute tpus queued-resources delete tpu3 --project raden-tpu  --zone europe-west4-b --force --async

gcloud compute tpus queued-resources ssh tpu3 --project raden-tpu --zone europe-west4-b
```