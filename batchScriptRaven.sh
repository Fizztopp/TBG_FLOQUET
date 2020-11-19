#!/bin/bash -x                                                                            
                                                                                          
#SBATCH -J timing                                                                         
                                                                                          
#SBATCH -N 1                                                                              
#SBATCH --ntasks 1                                                                 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
                                                                                          
###SBATCH --output=BATCH_OUT/timing.%j.out                                                  
###SBATCH --error=BATCH_OUT/timing-err.%j.out                                               
                                                                                          
#SBATCH --time=00:01:00     
#SBATCH --mem=4096
#SBATCH --partition=small                                                                 
                                                                                          
#SBATCH --mail-user=christian.eckhardt@rwth-aachen.de                                     
#SBATCH --mail-type=ALL                                                                   
                                                                                          
module purge
module load intel/19.1.3
module load hdf5-serial/1.8.21
module load mkl/2020.2
module load impi/2019.9

export OMP_NUM_THREADS=1                                                                  
export LD_LIBRARY_PATH="$MKL_HOME/lib/intel64_lin" 

srun ./TBG.out     
