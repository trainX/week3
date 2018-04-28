#Exercise 3 - Preparing for the project

##Goal: To give you experience with ingesting data via Nifi, storing it in the Hadoop File System (HDFS) and using Tensorflow 
to train the data.

##System Access:

The ip address will be provided via the trainX slack channel

Nifi at http://<ip address>:8080/nifi
Zeppelin is located at http://<ip address>:9995

##Task(s)

1. Ingest Detroit Crime data using the InvokeHTTP Processor.  The crime data can be found at https://data.detroitmi.gov/resource/9i6z-cm98.json.  
Details of the dataset can be found here: https://dev.socrata.com/foundry/data.detroitmi.gov/9i6z-cm98

2. The data should be stored into HDFS.  The HDFS location where that data should be stored is /data/<your name>-detroitcrimedata

3. Use Tensorflow to clean and extract 3 features of your chosing

4. Train the data using Tensorflow LinearRegressor (showed a basis example earlier this week)