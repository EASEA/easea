# A easea container
FROM pallamidessi/ubuntu-easea

MAINTAINER Pallamidessi Joseph version: 0.1

RUN apt-get update && cd && dpkg -i *.deb; exit 0 
RUN apt-get install -f -y
RUN cd /root && /bin/bash -c "source .bashrc"

