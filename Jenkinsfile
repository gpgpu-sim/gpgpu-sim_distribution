pipeline {
    agent {
        label "purdue-cluster"
        }

    stages {
        stage('4.2-regress') {
            steps {
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                source `pwd`/setup_environment && \
                make -j && \
                rm -rf gpgpu-sim_simulations && \
                git clone https://github.com/tgrogers/gpgpu-sim_simulations.git && \
                cd gpgpu-sim_simulations && \
                git checkout purdue-cluster && \
                make -j -C ./benchmarks/src all && \
                ./util/job_launching/run_simulations.py -N regress && \
                ./util/job_launching/monitor_func_test.py -v -N regress'

                emailex
                body: '''$JOB_NAME - $BRANCH_NAME - Build # currentBuild.number - currentBuild.result:
                Check console output at $BUILD_URL to view the results.''',
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                [$class: 'RequesterRecipientProvider']],
                replyTo: 'tgrogers@purdue.edu',
                subject: '[AALP Jenkins]: $JOB_NAME - Build # currentBuild.number - currentBuild.result!',
                to: 'tgrogers@purdue.edu'
            }
        }
    }



}
