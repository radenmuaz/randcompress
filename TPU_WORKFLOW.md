# tpu5
gcloud compute tpus queued-resources ssh tpu5 --project raden-tpu --zone us-central2-b --command="echo ok"
rm -f ~/.ssh/controlmasters/tpu5-muaz@35.186.33.7:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu5-%r@%h:%p \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/google_compute_engine muaz@35.186.33.7 "echo connected"

rm -f ~/.ssh/controlmasters/tpu5-muaz@35.186.33.7:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu5-%r@%h:%p \
  -i ~/.ssh/google_compute_engine muaz@35.186.33.7 "echo connected"

rsync -avz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='logs' --exclude='checkpoints' --exclude='.venv' \
  --exclude='datasets' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='.env' \
  -e "ssh -o ControlPath=$HOME/.ssh/controlmasters/tpu5-%r@%h:%p -i $HOME/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/ muaz@35.186.110.50:~/qcute/

# tpu6
gcloud compute tpus queued-resources ssh tpu6 --project raden-tpu --zone us-central2-b --command="echo ok"
rm -f ~/.ssh/controlmasters/tpu6-muaz@35.186.110.50:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu6-%r@%h:%p \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/google_compute_engine muaz@35.186.110.50 "echo connected"

rm -f ~/.ssh/controlmasters/tpu6-muaz@35.186.110.50:22
ssh -o ControlMaster=auto -o ControlPersist=600 \
  -o ControlPath=~/.ssh/controlmasters/tpu6-%r@%h:%p \
  -i ~/.ssh/google_compute_engine muaz@35.186.110.50 "echo connected"

rsync -avz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='logs' --exclude='checkpoints' --exclude='.venv' \
  --exclude='datasets' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='.env' \
  -e "ssh -o ControlPath=$HOME/.ssh/controlmasters/tpu6-%r@%h:%p -i $HOME/.ssh/google_compute_engine" \
  /Users/muaz/code/qcute/ muaz@35.186.110.50:~/qcute/

# configs only
echo 'scp -o ControlPath=$HOME/.ssh/controlmasters/tpu5-%r@%h:%p -i $HOME/.ssh/google_compute_engine \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py \
  muaz@35.186.33.7:~/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py

scp -o ControlPath=$HOME/.ssh/controlmasters/tpu6-%r@%h:%p -i $HOME/.ssh/google_compute_engine \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like.py \
  /Users/muaz/code/qcute/summformer_jax/image_classification/configs/tiny_vit_like_bidir_shared.py \
  muaz@35.186.110.50:~/qcute/summformer_jax/image_classification/configs/'