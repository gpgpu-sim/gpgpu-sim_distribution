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
    }
    post {
        success {
            emailext body:'''${SCRIPT, template="groovy-html.success.template"}''',
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
