pipeline {
    agent {
        label "purdue-cluster"
    }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('simulator-build') {
            steps {
                parallel "4.2": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    make -j'
                }, "9.1" : {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    make -j'
                }
            }
        }
        stage('simulations-build'){
            steps{
                sh 'rm -rf gpgpu-sim_simulations'
                sh 'git clone git@github.rcac.purdue.edu:TimRogersGroup/gpgpu-sim_simulations.git && \
                    cd gpgpu-sim_simulations && \
                    git checkout purdue-cluster && \
                    git pull && \
                    rm -r ./benchmarks/data_dirs && ln -s /home/tgrogers-raid/a/common/data_dirs benchmarks/'
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    cd gpgpu-sim_simulations && \
                    source ./benchmarks/src/setup_environment && \
                    make -j -C ./benchmarks/src rodinia_2.0-ft sdk-4.2 && \
                    make -C ./benchmarks/src data'
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    cd gpgpu-sim_simulations && \
                    source ./benchmarks/src/setup_environment && \
                    make -j -C ./benchmarks/src/ rodinia_2.0-ft sdk-4.2 && \
                    make -C ./benchmarks/src data'
            }
        }
        stage('regress'){
            steps {

                parallel "4.2-rodinia": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B rodinia_2.0-ft -C GTX480,GTX480-PTXPLUS -N regress-$$ && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/4.2-rodinia" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$ -s stats-$$.csv && \
                    ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR'
                }, "9.1-rodinia": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B rodinia_2.0-ft -C TITANX-P102,TITANX-P102-L1ON,P100-HBM -N regress-$$ && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/9.1-rodinia" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v  -s stats-$$.csv -N regress-$$ && \
                    ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR'
                }, "4.2-sdk-4.2": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B sdk-4.2 -C GTX480,TITANX-P102,TITANX-P102-L1ON,P100-HBM -N regress-$$ && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/4.2-sdk-4.2" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$ -s stats-$$.csv && \
                    ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR'
                }, "9.1-sdk-4.2": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B sdk-4.2 -C TITANX-P102,TITANX-P102-L1ON,P100-HBM -N regress-$$ && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/9.1-sdk-4.2" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$ -s stats-$$.csv && \
                    ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-$$.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR'
                }
            }
        }
        stage('4.2-correlate'){
            steps {
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/get_stats.py -R -K -k -B sdk-4.2,rodinia_2.0-ft -C TITANX-P102,P100-HBM > stats-4.2.csv && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/correlate-4.2" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    sh ./gpgpu-sim_simulations/run_hw/get_hw_data.sh && rm -rf ./gpgpu-sim_simulations/util/plotting/correl-html &&\
                    ./gpgpu-sim_simulations/util/plotting/plot-correlation.py -c stats-4.2.csv -H ./gpgpu-sim_simulations/run_hw/ &&\
                    scp  ./gpgpu-sim_simulations/util/plotting/correl-html/* tgrogers@dynamo.ecn.purdue.edu:/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR'
            }
        }
        stage('9.1-correlate'){
            steps {
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/get_stats.py -R -K -k -B sdk-4.2,rodinia_2.0-ft -C TITANX-P102,P100-HBM > stats-9.1.csv && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/correlate-9.1" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    sh ./gpgpu-sim_simulations/run_hw/get_hw_data.sh && rm -rf ./gpgpu-sim_simulations/util/plotting/correl-html &&\
                    ./gpgpu-sim_simulations/util/plotting/plot-correlation.py -c stats-9.1.csv -H ./gpgpu-sim_simulations/run_hw/ &&\
                    scp  ./gpgpu-sim_simulations/util/plotting/correl-html/* tgrogers@dynamo.ecn.purdue.edu:/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR'
            }
        }
    }
    post {
        success {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - Success!",
                to: 'tgrogers@purdue.edu'
        }
        failure {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'
        }
    }
}

