pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }
    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'jenkins_pipeline', url: 'https://github.com/RajeshB-0699/01_01_MLOPS_HOTEL_PRJ.git']])
                }
            }
        }

        stage('Setting up our venv and installing dependencies'){
            steps{
                script{
                    echo 'Setting up our venv and installing dependencies'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}
