#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=1:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/nanoGPT/logs/job_train_nonce_10k.err
#PJM -o /home/pj25000107/ku50001566/projects/nanoGPT/logs/job_train_nonce_10k.out
#PJM -N "10k_nonce_train"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/nanoGPT || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_nonce_10k.sh
