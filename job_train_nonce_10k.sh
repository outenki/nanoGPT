#!/bin/bash
#PJM -L "rscgrp=b-inter"
#PJM -L "elapse=1:00:00"
#PJM -L "gpu=3"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/nanoGPT/logs/job_generate_nonce_data.err
#PJM -o /home/pj25000107/ku50001566/projects/nanoGPT/logs/job_generate_nonce_data.out
#PJM -N "singularityjob"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/nanoGPT || exit 1

singularity exec /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_nonce_10k.sh
