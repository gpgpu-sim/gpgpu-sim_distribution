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
                    ln -s /home/tgrogers-raid/a/common/data_dirs benchmarks/'
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
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v -N regress-$$ -s stats-per-app-4.2.csv && \
                    ./gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-per-app-4.2.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR '
                }, "9.1-functest": {
                    sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    ./gpgpu-sim_simulations/util/job_launching/run_simulations.py -B rodinia_2.0-ft,sdk-4.2 -C TITANX,TITANX-L1ON,P100,TITANV -N regress-$$ && \
                    PLOTDIR="jenkins/${JOB_NAME}/${BUILD_NUMBER}/9.1-rodinia" && ssh tgrogers@dynamo.ecn.purdue.edu mkdir -p /home/dynamo/a/tgrogers/website/gpgpu-sim-plots/$PLOTDIR && \
                    ./gpgpu-sim_simulations/util/job_launching/monitor_func_test.py -v  -s stats-per-app-9.1.csv -N regress-$$ && \
                    ../gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c stats-per-app-9.1.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR'
                }
            }
        }
        stage('4.2-correlate'){
            steps {
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    PLOTDIR="jenkins/${JOB_NAME}" &&\
                    ./gpgpu-sim_simulations/util/job_launching/get_stats.py -R -K -k -B rodinia_2.0-ft -C GTX480,GTX480-PTXPLUS > stats-per-kernel-4.2.csv &&\
                    ./gpgpu-sim_simulations/util/plotting/correlate_and_publish.sh stats-per-kernel-4.2.csv $PLOTDIR ${BUILD_NUMBER}'
            }
        }
        stage('9.1-correlate'){
            steps {
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                    source `pwd`/setup_environment &&\
                    PLOTDIR="jenkins/${JOB_NAME}" &&\
                    ./gpgpu-sim_simulations/util/job_launching/get_stats.py -R -K -k -B rodinia_2.0-ft,sdk-4.2 -C TITANX,TITANX-L1ON,P100,TITANV > stats-per-kernel-9.1.csv &&\
                    ./gpgpu-sim_simulations/util/plotting/correlate_and_publish.sh stats-per-kernel-9.1.csv $PLOTDIR ${BUILD_NUMBER}'
            }
        }
        stage('archive-and-delta') {
            steps {
                    sh 'rm -rf gpgpu-sim-results-repo'
                    sh 'git clone git@github.com:purdue-aalp/gpgpu-sim-results-repo.git'

                    # Merge previous mainline with this new result on a per-app basis and plot everything
                    sh './gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/jenkins/AALP/gpgpu-sim_distribution/dev-purdue-integration/stats-per-app-4.2.csv,./stats-per-app-4.2.csv -R > per-app-merge-4.2.csv'
                    sh './gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/jenkins/AALP/gpgpu-sim_distribution/dev-purdue-integration/stats-per-app-9.1.csv,./stats-per-app-9.1.csv -R > per-app-merge-9.1.csv'
                    sh './gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c per-app-merge-4.2.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR/deltas -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR -n $PLOTDIR/deltas'
                    sh './gpgpu-sim_simulations/util/plotting/plot-get-stats.py -c per-app-merge-9.1.csv -p tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim-plots/$PLOTDIR/deltas -w https://engineering.purdue.edu/tgrogers/gpgpu-sim-plots/$PLOTDIR/deltas -n $PLOTDIR/deltas'

                    # Merge previous mailine with this new result on a per-kernel basis and correlate.
                    sh './gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/jenkins/AALP/gpgpu-sim_distribution/dev-purdue-integration/stats-per-kernel-4.2.csv,./stats-per-kernel-4.2.csv -R > per-kernel-merge-4.2.csv'
                    sh './gpgpu-sim_simulations/util/plotting/merge-stats.py -c ./gpgpu-sim-results-repo/jenkins/AALP/gpgpu-sim_distribution/dev-purdue-integration/stats-per-kernel-9.1.csv,./stats-per-kernel-9.1.csv -R > per-kernel-merge-9.1.csv'
                    sh ' ./gpgpu-sim_simulations/util/plotting/correlate_and_publish.sh per-kernel-merge-4.2.csv $PLOTDIR ${BUILD_NUMBER} &&\
                         ./gpgpu-sim_simulations/util/plotting/correlate_and_publish.sh per-kernel-merge-9.1.csv $PLOTDIR ${BUILD_NUMBER}'


                    sh 'mkdir -p ./jenkins/quick-regress/${JOB_NAME}/ && cp stats-per-*.csv ./jenkins/quick-regress/${JOB_NAME}/'
                    sh 'cd ./gpgpu-sim-results-repo &&\
                        git add ./jenkins/quick-regress/${JOB_NAME}/* &&\
                        git commit -m "Jenkins automated checkin ${BUILD_NUMBER} && git push'
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

