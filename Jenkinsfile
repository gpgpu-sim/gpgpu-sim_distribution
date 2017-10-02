pipeline {
    agent {
        label "purdue-cluster"
    }

    stages {
        stage('simulator-build') {
            steps {
                
                parallel "4.2": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    make -j'
                }, "8.0" : {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/8.0_env_setup.sh &&\
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
                    git pull'
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    cd gpgpu-sim_simulations && \
                    source ./benchmarks/src/setup_environment && \
                    make -C ./benchmarks/src all'
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/8.0_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    cd gpgpu-sim_simulations && \
                    source ./benchmarks/src/setup_environment && \
                    make -f Makefile.PTX5 -C ./benchmarks/src/ all'
            }
        }
        stage('rodinia-regress'){
            steps {
                parallel "4.2-rodinia": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -b ./gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/benchmarks.yml -c ./gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/configs.yml -N regress-$$ && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$'
                }, "8.0-rodinia": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/8.0_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -b ./gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/benchmarks-8.0.yml -c ./gpgpu-sim_simulations/util/job_launching/regression_recipies/rodinia_2.0-ft/configs-fermi-plus-only.yml -N regress-$$ && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$'
                }
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

