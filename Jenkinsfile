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


                step(def to = emailextrecipients([
                    [$class: 'CulpritsRecipientProvider'],
                    [$class: 'DevelopersRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']
                ])

                if(to != null && !to.isEmpty()) {
                        mail to: to, subject: "[AALP Jenkins] Build ${currentBuild.result}!",
                                    body: "See ${env.BUILD_URL}"
                })
            }
        }
    }



}
