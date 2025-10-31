pipeline{
    agent any
    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'jenkins_pipeline', url: 'https://github.com/RajeshB-0699/01_01_MLOPS_HOTEL_PRJ.git']])
                }
            }
        }
    }
}
