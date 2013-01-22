#!/bin/bash
export LCG_GFAL_INFOSYS=topbdii.grif.fr:2170
export LCG_CATALOG_TYPE=lfc
export LCG_GFAL_VO=vo.complex-systems.eu

chmod +x $PWD/regression
$PWD/regression --nbGen=20

