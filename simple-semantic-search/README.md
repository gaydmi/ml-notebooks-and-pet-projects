# Text classifier based on semantic proximity

## Launch contrainer instructions
Build this server in docker container
```
docker build -t predict_task:latest .
```
Start running the server
```
docker run -d -p 5000:5000 predict_task
```
Stop and remove all the containers launched in docker
```
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
```

To test API of the server you could launch client_testing.py
