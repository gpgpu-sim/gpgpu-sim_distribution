pipeline {
    agent {
        label "purdue-cluster"
    }

    options {
        disableConcurrentBuilds()
    }
    stages {
        /*
        stage('formatting-check') {
          steps {
            sh '''
              source ./env-setup/common/export_gcc_version.sh 5.3.0
              git remote add upstream https://github.com/purdue-aalp/gpgpu-sim_distribution
              git fetch upstream
              if git diff --name-only upstream/dev | grep -E "*.cc|*.h|*.cpp|*.hpp" ; then
                git diff --name-only upstream/dev | grep -E "*.cc|*.h|*.cpp|*.hpp" | xargs ./run-clang-format.py --clang-format-executable /home/tgrogers-raid/a/common/clang-format/6.0.1/clang-format
              fi
            ''' 
          }
        }
        */
        stage('env-setup') {
            steps {
                sh 'rm -rf env-setup && git clone git@github.com:purdue-aalp/env-setup.git &&\
                    cd env-setup && git checkout cluster-ubuntu'
            }
        }
        stage('simulator-build') {
            steps {
                sh '''#!/bin/bash
                    source ./env-setup/11.0_env_setup.sh
                    source `pwd`/setup_environment
                    make -j 10'''
            }
        }
        stage('simulations-build'){
            steps{
                sh 'rm -rf gpgpu-sim_simulations'
                sh 'git clone git@github.com:purdue-aalp/gpgpu-sim_simulations.git && \
                    cd gpgpu-sim_simulations && \
                    git pull && \
                    ln -s /home/tgrogers-raid/a/common/data_dirs benchmarks/'
                sh '''#!/bin/bash
                    source ./env-setup/11.0_env_setup.sh
                    source `pwd`/setup_environment
                    cd gpgpu-sim_simulations
                    source ./benchmarks/src/setup_environment
                    make -j 10 -C ./benchmarks/src/ rodinia_2.0-ft
                    make -C ./benchmarks/src data'''
            }
        }
        stage('11.0 Regressions'){
            steps {
                    sh '''#!/bin/bash
                        source ./env-setup/11.0_env_setup.sh
                        source `pwd`/setup_environment
                        ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100 -N regress-$$ 
                        PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/11.0" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR
                        ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -s stats-per-app-11.0.csv -N regress-$$'''
            }
        }
        stage('correlate-delta-and-archive') {
            steps {
                    sh './gpgpu-sim_simulations/run_hw/get_hw_data.sh'
                    sh 'rm -rf ./gpgpu-sim_simulations/util/plotting/correl-html && rm -rf gpgpu-sim-results-repo && rm -rf ./gpgpu-sim_simulations/util/plotting/htmls'
                    sh 'git clone git@github.com:purdue-aalp/gpgpu-sim-results-repo.git'
                    sh '''#!/bin/bash
                        source ./env-setup/11.0_env_setup.sh
                        ./gpgpu-sim_simulations/util/job_launching/get_stats.py -R -K -k -B rodinia_2.0-ft -C QV100 -A > stats-per-kernel-11.0.csv'''
                    sh 'if [ ! -d ./gpgpu-sim-results-repo/${JOB_NAME} ]; then mkdir -p ./gpgpu-sim-results-repo/${JOB_NAME}/ ; cp ./gpgpu-sim-results-repo/purdue-aalp/gpgpu-sim_distribution/dev/* ./gpgpu-sim-results-repo/${JOB_NAME}/ ; fi'
                    sh './gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/${JOB_NAME}/stats-per-app-11.0.csv,./stats-per-app-11.0.csv -R > per-app-merge-11.0.csv'
                    sh 'PLOTDIR="jenkins/${JOB_NAME}" &&\
                        ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c per-app-merge-11.0.csv -P cuda-11.0 &&\
                        ./gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/${JOB_NAME}/stats-per-kernel-11.0.csv,./stats-per-kernel-11.0.csv -R > per-kernel-merge-11.0.csv &&\
                        ./gpgpu-sim_simulations/util/plotting/plot-correlation.py -H ./gpgpu-sim_simulations/run_hw/QUADRO-V100/device-0/9.1/ -c per-kernel-merge-11.0.csv -p cuda-11.0 | grep -B 1 "Correl=" | tee correl.11.0.txt &&\
                        mkdir -p ./gpgpu-sim-results-repo/${JOB_NAME}/ && cp stats-per-*.csv ./gpgpu-sim-results-repo/${JOB_NAME}/ &&\
                        cd ./gpgpu-sim-results-repo &&\
                        git diff --quiet && git diff --staged --quiet || git commit -am "Jenkins automated checkin ${JOB_NAME} Build:${BUILD_NUMBER}" &&\
                        git push'

                    sh 'PLOTDIR="/home/dynamo/a/tgrogers/website/gpgpu-sim-plots/jenkins/${JOB_NAME}" &&\
                        ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p $PLOTDIR/${BUILD_NUMBER} && \
                        scp  ./gpgpu-sim_simulations/util/plotting/correl-html/* tgrogers@dynamo.ecn.purdue.edu:$PLOTDIR/${BUILD_NUMBER} &&\
                        scp  ./gpgpu-sim_simulations/util/plotting/htmls/* tgrogers@dynamo.ecn.purdue.edu:$PLOTDIR/${BUILD_NUMBER} &&\
                        ssh tgrogers@dynamo.ecn.purdue.edu "cd $PLOTDIR && rm -rf latest && cp -r ${BUILD_NUMBER} latest"'
            }
        }
    }
    post {
        success {
//            sh 'git remote rm upstream'
            emailext body:'''${SCRIPT, template="groovy-html.success.template"}''',
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - Success!",
                attachmentsPattern: 'correl.*.txt',
                to: 'tgrogers@purdue.edu'
        }
        failure {
//            sh 'git remote rm upstream'
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'
        }
    }
}
