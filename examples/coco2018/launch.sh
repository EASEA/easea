#!/bin/bash

echo "Launch COCO2018...."
./coco2018
echo "Start plot script..."
python -m cocopp exdata
echo "Done"
