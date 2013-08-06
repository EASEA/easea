#!/bin/bash
export LCG_GFAL_INFOSYS=topbdii.grif.fr:2170
export LCG_CATALOG_TYPE=lfc
export LCG_GFAL_VO=vo.complex-systems.eu
#export X509_USER_PROXY=x509up_u500
#tar --version
lcg-cp lfn:/grid/vo.complex-systems.eu/easea/cde-package.tar.gz cde-package.tar.gz
tar -xvf cde-package.tar.gz
cd cde-package/cde-root/home/ge-user/
tar -xvf ../../../../application.tar.gz
wget -qO- ifconfig.me/ip>external_ip.txt
../../../cde-exec ./APPLICATIONNAME --worker_number=$1

